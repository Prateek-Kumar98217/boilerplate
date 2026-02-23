/**
 * StripeProvider — implements PaymentProvider using Stripe Checkout.
 *
 * Install: npm install stripe
 *
 * Env vars:
 *   STRIPE_SECRET_KEY=sk_...
 *   STRIPE_WEBHOOK_SECRET=whsec_...
 */
import Stripe from "stripe";
import { paymentOk, paymentError } from "./types";
import type {
  PaymentProvider,
  PaymentResult,
  CheckoutSession,
  CreateCheckoutInput,
  CreateCustomerInput,
  CustomerRecord,
  CreateSubscriptionInput,
  SubscriptionRecord,
  RefundRecord,
  CreateRefundInput,
  WebhookPayload,
} from "./types";

// ─── Client ───────────────────────────────────────────────────────────────────

function buildStripeClient(): Stripe {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) {
    throw new Error("STRIPE_SECRET_KEY environment variable is not set.");
  }
  return new Stripe(key, { apiVersion: "2025-01-27.acacia" });
}

// ─── Normalise Stripe errors ─────────────────────────────────────────────────

function handleStripeError(err: unknown): PaymentResult<never> {
  if (err instanceof Stripe.errors.StripeCardError) {
    const code =
      err.code === "card_declined"
        ? "CARD_DECLINED"
        : err.code === "insufficient_funds"
          ? "INSUFFICIENT_FUNDS"
          : "PROVIDER_ERROR";
    return paymentError(err.message, code);
  }
  if (err instanceof Stripe.errors.StripeInvalidRequestError) {
    return paymentError(err.message, "INVALID_INPUT");
  }
  if (err instanceof Stripe.errors.StripeAPIError) {
    return paymentError(err.message, "PROVIDER_ERROR");
  }
  if (err instanceof Stripe.errors.StripeConnectionError) {
    return paymentError("Could not connect to Stripe.", "NETWORK_ERROR");
  }
  const message = err instanceof Error ? err.message : "Unknown Stripe error.";
  return paymentError(message, "UNKNOWN");
}

// ─── Event normalisation ─────────────────────────────────────────────────────

function normaliseStripeEvent(event: Stripe.Event): WebhookPayload | null {
  switch (event.type) {
    case "payment_intent.succeeded": {
      const pi = event.data.object as Stripe.PaymentIntent;
      return {
        event: "payment.succeeded",
        paymentId: pi.id,
        orderId: pi.metadata["orderId"] ?? pi.id,
        amount: pi.amount,
        currency: pi.currency,
        metadata: pi.metadata,
      };
    }
    case "payment_intent.payment_failed": {
      const pi = event.data.object as Stripe.PaymentIntent;
      return {
        event: "payment.failed",
        paymentId: pi.id,
        orderId: pi.metadata["orderId"] ?? pi.id,
        reason: pi.last_payment_error?.message ?? "Unknown reason",
      };
    }
    case "customer.subscription.updated":
    case "customer.subscription.created": {
      const sub = event.data.object as Stripe.Subscription;
      if (sub.status === "active" || sub.status === "trialing") {
        return {
          event: "subscription.activated",
          subscriptionId: sub.id,
          customerId:
            typeof sub.customer === "string" ? sub.customer : sub.customer.id,
        };
      }
      if (sub.status === "past_due") {
        return {
          event: "subscription.past_due",
          subscriptionId: sub.id,
          customerId:
            typeof sub.customer === "string" ? sub.customer : sub.customer.id,
        };
      }
      return null;
    }
    case "customer.subscription.deleted": {
      const sub = event.data.object as Stripe.Subscription;
      return {
        event: "subscription.cancelled",
        subscriptionId: sub.id,
        customerId:
          typeof sub.customer === "string" ? sub.customer : sub.customer.id,
      };
    }
    case "charge.refunded": {
      const charge = event.data.object as Stripe.Charge;
      const refund = charge.refunds?.data[0];
      if (!refund) return null;
      return {
        event: "refund.created",
        refundId: refund.id,
        paymentId: charge.id,
        amount: refund.amount,
      };
    }
    default:
      return null;
  }
}

// ─── Implementation ───────────────────────────────────────────────────────────

export class StripeProvider implements PaymentProvider {
  readonly name = "stripe" as const;
  private readonly stripe: Stripe;

  constructor() {
    this.stripe = buildStripeClient();
  }

  // ── Checkout ────────────────────────────────────────────────────────────────

  async createCheckout(
    input: CreateCheckoutInput,
  ): Promise<PaymentResult<CheckoutSession>> {
    try {
      const session = await this.stripe.checkout.sessions.create({
        payment_method_types: ["card"],
        mode: "payment",
        customer_email: input.customerEmail,
        ...(input.customerId ? { customer: input.customerId } : {}),
        line_items: [
          {
            quantity: 1,
            price_data: {
              currency: input.currency,
              unit_amount: input.amount,
              product_data: {
                name: input.description ?? "Order",
              },
            },
          },
        ],
        success_url: input.successUrl,
        cancel_url: input.cancelUrl,
        client_reference_id: input.orderId,
        metadata: { orderId: input.orderId, ...(input.metadata ?? {}) },
      });

      if (!session.url) {
        return paymentError(
          "Stripe returned a session without a URL.",
          "PROVIDER_ERROR",
        );
      }

      return paymentOk({
        checkoutUrl: session.url,
        providerSessionId: session.id,
        expiresAt: session.expires_at
          ? new Date(session.expires_at * 1000).toISOString()
          : undefined,
      });
    } catch (err) {
      return handleStripeError(err);
    }
  }

  // ── Customers ───────────────────────────────────────────────────────────────

  async createCustomer(
    input: CreateCustomerInput,
  ): Promise<PaymentResult<CustomerRecord>> {
    try {
      const customer = await this.stripe.customers.create({
        email: input.email,
        name: input.name,
        phone: input.phone,
        metadata: input.metadata,
      });
      return paymentOk({
        providerCustomerId: customer.id,
        email: customer.email ?? input.email,
        name: customer.name ?? undefined,
      });
    } catch (err) {
      return handleStripeError(err);
    }
  }

  // ── Subscriptions ────────────────────────────────────────────────────────────

  async createSubscription(
    input: CreateSubscriptionInput,
  ): Promise<PaymentResult<SubscriptionRecord>> {
    try {
      const sub = await this.stripe.subscriptions.create({
        customer: input.customerId,
        items: [{ price: input.planId }],
        trial_end: input.trialEndDate
          ? Math.floor(new Date(input.trialEndDate).getTime() / 1000)
          : undefined,
        metadata: input.metadata,
        payment_behavior: "default_incomplete",
        expand: ["latest_invoice.payment_intent"],
      });

      return paymentOk({
        providerSubscriptionId: sub.id,
        status: sub.status,
        currentPeriodEnd: new Date(sub.current_period_end * 1000).toISOString(),
      });
    } catch (err) {
      return handleStripeError(err);
    }
  }

  async cancelSubscription(
    subscriptionId: string,
  ): Promise<PaymentResult<{ cancelled: true }>> {
    try {
      await this.stripe.subscriptions.cancel(subscriptionId);
      return paymentOk({ cancelled: true as const });
    } catch (err) {
      return handleStripeError(err);
    }
  }

  // ── Refunds ──────────────────────────────────────────────────────────────────

  async createRefund(
    input: CreateRefundInput,
  ): Promise<PaymentResult<RefundRecord>> {
    try {
      const refund = await this.stripe.refunds.create({
        payment_intent: input.paymentId,
        ...(input.amount !== undefined ? { amount: input.amount } : {}),
        reason: (input.reason as Stripe.RefundCreateParams.Reason) ?? undefined,
      });
      return paymentOk({
        providerRefundId: refund.id,
        status: refund.status ?? "unknown",
        amount: refund.amount,
      });
    } catch (err) {
      return handleStripeError(err);
    }
  }

  // ── Webhooks ─────────────────────────────────────────────────────────────────

  async parseWebhook(
    rawBody: string,
    signature: string,
  ): Promise<PaymentResult<WebhookPayload>> {
    const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
    if (!webhookSecret) {
      return paymentError(
        "STRIPE_WEBHOOK_SECRET is not set.",
        "WEBHOOK_SIGNATURE_INVALID",
      );
    }

    let event: Stripe.Event;
    try {
      event = this.stripe.webhooks.constructEvent(
        rawBody,
        signature,
        webhookSecret,
      );
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Signature verification failed.";
      return paymentError(msg, "WEBHOOK_SIGNATURE_INVALID");
    }

    const payload = normaliseStripeEvent(event);
    if (!payload) {
      return paymentError(`Unhandled event type: ${event.type}`, "UNKNOWN");
    }

    return paymentOk(payload);
  }
}

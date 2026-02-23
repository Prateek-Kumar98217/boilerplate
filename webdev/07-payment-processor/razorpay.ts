/**
 * RazorpayProvider — implements PaymentProvider using Razorpay Orders + Payment Links.
 *
 * Install: npm install razorpay
 *
 * Env vars:
 *   RAZORPAY_KEY_ID=rzp_...
 *   RAZORPAY_KEY_SECRET=...
 *   RAZORPAY_WEBHOOK_SECRET=...
 *
 * Notes:
 *  - Razorpay works with the smallest currency unit (paise for INR, US cents for USD).
 *  - Hosted checkout is a Payment Link; the `checkoutUrl` is the short URL.
 *  - Subscriptions require a pre-configured Plan in the Razorpay dashboard.
 */
import Razorpay from "razorpay";
import { createHmac } from "node:crypto";
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

// ─── Razorpay SDK types are not always great; define minimal shapes here ─────

interface RazorpayOrder {
  id: string;
  amount: number;
  currency: string;
  receipt?: string;
  notes?: Record<string, string>;
}

interface RazorpayCustomer {
  id: string;
  email: string;
  name: string;
  contact?: string;
}

interface RazorpaySubscription {
  id: string;
  status: string;
  current_end: number;
}

interface RazorpayPaymentLink {
  id: string;
  short_url: string;
  expire_by?: number;
}

interface RazorpayRefund {
  id: string;
  amount: number;
  status: string;
}

// ─── Client ───────────────────────────────────────────────────────────────────

function buildRazorpayClient(): Razorpay {
  const keyId = process.env.RAZORPAY_KEY_ID;
  const keySecret = process.env.RAZORPAY_KEY_SECRET;
  if (!keyId || !keySecret) {
    throw new Error(
      "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET environment variables are required.",
    );
  }
  return new Razorpay({ key_id: keyId, key_secret: keySecret });
}

// ─── Normalise Razorpay errors ────────────────────────────────────────────────

function handleRazorpayError(err: unknown): PaymentResult<never> {
  if (err && typeof err === "object" && "error" in err) {
    const rErr = (err as { error: { description: string; code?: string } })
      .error;
    return paymentError(rErr.description ?? "Razorpay error", "PROVIDER_ERROR");
  }
  const message =
    err instanceof Error ? err.message : "Unknown Razorpay error.";
  return paymentError(message, "UNKNOWN");
}

// ─── Webhook payload normalisation ────────────────────────────────────────────

function normaliseRazorpayEvent(
  event: string,
  payload: Record<string, unknown>,
): WebhookPayload | null {
  switch (event) {
    case "payment.captured": {
      const p = (payload as { payment: { entity: Record<string, unknown> } })
        .payment.entity;
      return {
        event: "payment.succeeded",
        paymentId: p["id"] as string,
        orderId: (p["order_id"] as string) ?? (p["id"] as string),
        amount: p["amount"] as number,
        currency: p["currency"] as string,
        metadata: (p["notes"] as Record<string, string>) ?? {},
      };
    }
    case "payment.failed": {
      const p = (payload as { payment: { entity: Record<string, unknown> } })
        .payment.entity;
      return {
        event: "payment.failed",
        paymentId: p["id"] as string,
        orderId: (p["order_id"] as string) ?? (p["id"] as string),
        reason: (p["error_description"] as string) ?? "Unknown reason",
      };
    }
    case "subscription.activated": {
      const s = (
        payload as { subscription: { entity: Record<string, unknown> } }
      ).subscription.entity;
      return {
        event: "subscription.activated",
        subscriptionId: s["id"] as string,
        customerId: (s["customer_id"] as string) ?? "",
      };
    }
    case "subscription.halted":
    case "subscription.pending": {
      const s = (
        payload as { subscription: { entity: Record<string, unknown> } }
      ).subscription.entity;
      return {
        event: "subscription.past_due",
        subscriptionId: s["id"] as string,
        customerId: (s["customer_id"] as string) ?? "",
      };
    }
    case "subscription.cancelled": {
      const s = (
        payload as { subscription: { entity: Record<string, unknown> } }
      ).subscription.entity;
      return {
        event: "subscription.cancelled",
        subscriptionId: s["id"] as string,
        customerId: (s["customer_id"] as string) ?? "",
      };
    }
    case "refund.created": {
      const r = (payload as { refund: { entity: Record<string, unknown> } })
        .refund.entity;
      return {
        event: "refund.created",
        refundId: r["id"] as string,
        paymentId: r["payment_id"] as string,
        amount: r["amount"] as number,
      };
    }
    default:
      return null;
  }
}

// ─── Implementation ───────────────────────────────────────────────────────────

export class RazorpayProvider implements PaymentProvider {
  readonly name = "razorpay" as const;
  private readonly client: Razorpay;

  constructor() {
    this.client = buildRazorpayClient();
  }

  // ── Checkout ─────────────────────────────────────────────────────────────────
  // Razorpay hosted checkout = Payment Link.
  // For embedded checkout, create an Order and use Razorpay.js client-side.

  async createCheckout(
    input: CreateCheckoutInput,
  ): Promise<PaymentResult<CheckoutSession>> {
    try {
      // First create an order so we have a stable reference.
      const order = (await (
        this.client.orders as unknown as {
          create(data: Record<string, unknown>): Promise<RazorpayOrder>;
        }
      ).create({
        amount: input.amount,
        currency: input.currency.toUpperCase(),
        receipt: input.orderId,
        notes: { orderId: input.orderId, ...input.metadata },
      })) as RazorpayOrder;

      // Create a Payment Link that wraps the order.
      const link = (await (
        this.client.paymentLink as unknown as {
          create(data: Record<string, unknown>): Promise<RazorpayPaymentLink>;
        }
      ).create({
        amount: input.amount,
        currency: input.currency.toUpperCase(),
        description: input.description ?? "Order",
        customer: {
          email: input.customerEmail,
        },
        notify: { email: true },
        reminder_enable: false,
        callback_url: input.successUrl,
        callback_method: "get",
        notes: { orderId: order.id, ...input.metadata },
      })) as RazorpayPaymentLink;

      return paymentOk({
        checkoutUrl: link.short_url,
        providerSessionId: link.id,
        expiresAt: link.expire_by
          ? new Date(link.expire_by * 1000).toISOString()
          : undefined,
      });
    } catch (err) {
      return handleRazorpayError(err);
    }
  }

  // ── Customers ─────────────────────────────────────────────────────────────────

  async createCustomer(
    input: CreateCustomerInput,
  ): Promise<PaymentResult<CustomerRecord>> {
    try {
      const customer = (await (
        this.client.customers as unknown as {
          create(data: Record<string, unknown>): Promise<RazorpayCustomer>;
        }
      ).create({
        name: input.name,
        email: input.email,
        contact: input.phone,
        fail_existing: 0, // do not fail if email already exists — return existing
        notes: input.metadata,
      })) as RazorpayCustomer;

      return paymentOk({
        providerCustomerId: customer.id,
        email: customer.email,
        name: customer.name,
      });
    } catch (err) {
      return handleRazorpayError(err);
    }
  }

  // ── Subscriptions ─────────────────────────────────────────────────────────────
  // Requires a Plan created in the Razorpay dashboard; input.planId = plan_...

  async createSubscription(
    input: CreateSubscriptionInput,
  ): Promise<PaymentResult<SubscriptionRecord>> {
    try {
      const sub = (await (
        this.client.subscriptions as unknown as {
          create(data: Record<string, unknown>): Promise<RazorpaySubscription>;
        }
      ).create({
        plan_id: input.planId,
        customer_id: input.customerId,
        total_count: 12, // number of billing cycles; adjust or expose as input
        start_at: input.trialEndDate
          ? Math.floor(new Date(input.trialEndDate).getTime() / 1000)
          : undefined,
        notes: input.metadata,
      })) as RazorpaySubscription;

      return paymentOk({
        providerSubscriptionId: sub.id,
        status: sub.status,
        currentPeriodEnd: new Date(sub.current_end * 1000).toISOString(),
      });
    } catch (err) {
      return handleRazorpayError(err);
    }
  }

  async cancelSubscription(
    subscriptionId: string,
  ): Promise<PaymentResult<{ cancelled: true }>> {
    try {
      await (
        this.client.subscriptions as unknown as {
          cancel(id: string): Promise<void>;
        }
      ).cancel(subscriptionId);
      return paymentOk({ cancelled: true as const });
    } catch (err) {
      return handleRazorpayError(err);
    }
  }

  // ── Refunds ───────────────────────────────────────────────────────────────────

  async createRefund(
    input: CreateRefundInput,
  ): Promise<PaymentResult<RefundRecord>> {
    try {
      const refund = (await (
        this.client.payments as unknown as {
          refund(
            paymentId: string,
            data: Record<string, unknown>,
          ): Promise<RazorpayRefund>;
        }
      ).refund(input.paymentId, {
        amount: input.amount, // if undefined, full refund
        notes: input.reason ? { reason: input.reason } : undefined,
      })) as RazorpayRefund;

      return paymentOk({
        providerRefundId: refund.id,
        status: refund.status,
        amount: refund.amount,
      });
    } catch (err) {
      return handleRazorpayError(err);
    }
  }

  // ── Webhooks ──────────────────────────────────────────────────────────────────
  // Razorpay signs webhooks with HMAC-SHA256; the header is "x-razorpay-signature".

  async parseWebhook(
    rawBody: string,
    signature: string,
  ): Promise<PaymentResult<WebhookPayload>> {
    const secret = process.env.RAZORPAY_WEBHOOK_SECRET;
    if (!secret) {
      return paymentError(
        "RAZORPAY_WEBHOOK_SECRET is not set.",
        "WEBHOOK_SIGNATURE_INVALID",
      );
    }

    // Verify signature
    const expected = createHmac("sha256", secret).update(rawBody).digest("hex");

    if (expected !== signature) {
      return paymentError(
        "Webhook signature mismatch.",
        "WEBHOOK_SIGNATURE_INVALID",
      );
    }

    let body: { event: string; payload: Record<string, unknown> };
    try {
      body = JSON.parse(rawBody) as typeof body;
    } catch {
      return paymentError("Webhook body is not valid JSON.", "UNKNOWN");
    }

    const webhookPayload = normaliseRazorpayEvent(body.event, body.payload);
    if (!webhookPayload) {
      return paymentError(`Unhandled Razorpay event: ${body.event}`, "UNKNOWN");
    }

    return paymentOk(webhookPayload);
  }
}

/**
 * PaymentProvider interface — the single contract both Stripe and Razorpay implement.
 *
 * Every method returns a discriminated union so callers can narrow without
 * try/catch and without knowledge of which provider is active.
 */

// ─── Shared value types ───────────────────────────────────────────────────────

/** ISO 4217 currency code, lowercase. e.g. "usd", "inr" */
export type CurrencyCode = string;

/** Amount in the currency's smallest unit. $10.00 = 1000, ₹500 = 50000 */
export type AmountInSmallestUnit = number;

// ─── Result types ─────────────────────────────────────────────────────────────

export type PaymentResult<T> =
  | { success: true; data: T }
  | { success: false; error: string; code: PaymentErrorCode };

export type PaymentErrorCode =
  | "INVALID_INPUT"
  | "PROVIDER_ERROR"
  | "NETWORK_ERROR"
  | "CARD_DECLINED"
  | "INSUFFICIENT_FUNDS"
  | "ALREADY_CANCELLED"
  | "NOT_FOUND"
  | "WEBHOOK_SIGNATURE_INVALID"
  | "UNKNOWN";

// ─── Checkout ─────────────────────────────────────────────────────────────────

export type CreateCheckoutInput = {
  /** Unique ID in your system — used as idempotency key. */
  orderId: string;
  customerId?: string;
  customerEmail?: string;
  customerName?: string;
  /** Amount in the currency's smallest unit. */
  amount: AmountInSmallestUnit;
  currency: CurrencyCode;
  /** Brief description shown to the user on the checkout page. */
  description?: string;
  /** URL to redirect to on successful payment. */
  successUrl: string;
  /** URL to redirect to on cancellation. */
  cancelUrl: string;
  /** Arbitrary key-value pairs forwarded to the payment provider (e.g. userId). */
  metadata?: Record<string, string>;
};

export type CheckoutSession = {
  /** The URL to redirect the customer to. */
  checkoutUrl: string;
  /** Provider's session/order ID — store this for webhook reconciliation. */
  providerSessionId: string;
  /** ISO timestamp when the session expires (if the provider supplies it). */
  expiresAt?: string;
};

// ─── Subscription management ──────────────────────────────────────────────────

export type CreateSubscriptionInput = {
  customerId: string;
  /** Provider-specific plan/price ID. */
  planId: string;
  /** ISO date string to start the trial, if any. */
  trialEndDate?: string;
  metadata?: Record<string, string>;
};

export type SubscriptionRecord = {
  providerSubscriptionId: string;
  status: string;
  currentPeriodEnd: string;
  checkoutUrl?: string; // Razorpay requires a hosted page for initial payment
};

// ─── Customer management ──────────────────────────────────────────────────────

export type CreateCustomerInput = {
  email: string;
  name?: string;
  phone?: string;
  metadata?: Record<string, string>;
};

export type CustomerRecord = {
  providerCustomerId: string;
  email: string;
  name?: string;
};

// ─── Refund ───────────────────────────────────────────────────────────────────

export type CreateRefundInput = {
  /** Provider's payment/charge ID. */
  paymentId: string;
  /** Optional partial refund amount in smallest unit. Omit for full refund. */
  amount?: AmountInSmallestUnit;
  reason?: string;
};

export type RefundRecord = {
  providerRefundId: string;
  status: string;
  amount: AmountInSmallestUnit;
};

// ─── Webhook ──────────────────────────────────────────────────────────────────

export type WebhookPayload =
  | {
      event: "payment.succeeded";
      paymentId: string;
      orderId: string;
      amount: AmountInSmallestUnit;
      currency: CurrencyCode;
      metadata: Record<string, string>;
    }
  | {
      event: "payment.failed";
      paymentId: string;
      orderId: string;
      reason: string;
    }
  | {
      event: "subscription.activated";
      subscriptionId: string;
      customerId: string;
    }
  | {
      event: "subscription.cancelled";
      subscriptionId: string;
      customerId: string;
    }
  | {
      event: "subscription.past_due";
      subscriptionId: string;
      customerId: string;
    }
  | {
      event: "refund.created";
      refundId: string;
      paymentId: string;
      amount: AmountInSmallestUnit;
    };

// ─── Provider interface ───────────────────────────────────────────────────────

export interface PaymentProvider {
  readonly name: "stripe" | "razorpay";

  // ── Checkout ────────────────────────────────────────────────────────────────

  /**
   * Creates a hosted checkout session and returns a `checkoutUrl` to redirect
   * the customer to. This is the primary integration point.
   */
  createCheckout(
    input: CreateCheckoutInput,
  ): Promise<PaymentResult<CheckoutSession>>;

  // ── Customers ───────────────────────────────────────────────────────────────

  createCustomer(
    input: CreateCustomerInput,
  ): Promise<PaymentResult<CustomerRecord>>;

  // ── Subscriptions ───────────────────────────────────────────────────────────

  createSubscription(
    input: CreateSubscriptionInput,
  ): Promise<PaymentResult<SubscriptionRecord>>;

  cancelSubscription(
    subscriptionId: string,
  ): Promise<PaymentResult<{ cancelled: true }>>;

  // ── Refunds ─────────────────────────────────────────────────────────────────

  createRefund(input: CreateRefundInput): Promise<PaymentResult<RefundRecord>>;

  // ── Webhooks ─────────────────────────────────────────────────────────────────

  /**
   * Verifies the webhook signature and parses the raw body into a
   * normalised WebhookPayload. Returns an error result if the signature
   * is invalid.
   */
  parseWebhook(
    rawBody: string,
    signature: string,
  ): Promise<PaymentResult<WebhookPayload>>;
}

// ─── Error helper ─────────────────────────────────────────────────────────────

export function paymentError(
  error: string,
  code: PaymentErrorCode = "UNKNOWN",
): PaymentResult<never> {
  return { success: false, error, code };
}

export function paymentOk<T>(data: T): PaymentResult<T> {
  return { success: true, data };
}

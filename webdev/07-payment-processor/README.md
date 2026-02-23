# 07 · Payment Processor

A typed `PaymentProvider` interface with Stripe and Razorpay implementations.
Both providers return the same shape — swap them with one line.

---

## File map

```
07-payment-processor/
├── types.ts                         Shared types, PaymentProvider interface
├── stripe.ts                        StripeProvider implements PaymentProvider
├── razorpay.ts                      RazorpayProvider implements PaymentProvider
├── index.ts                         Barrel + createPaymentProvider() factory
└── examples/
    ├── stripe-webhook-route.ts      Next.js route handler for Stripe webhooks
    └── razorpay-webhook-route.ts    Next.js route handler for Razorpay webhooks
```

---

## Core interface

```ts
interface PaymentProvider {
  readonly name: "stripe" | "razorpay";
  createCheckout(input): Promise<PaymentResult<CheckoutSession>>;
  createCustomer(input): Promise<PaymentResult<CustomerRecord>>;
  createSubscription(input): Promise<PaymentResult<SubscriptionRecord>>;
  cancelSubscription(id): Promise<PaymentResult<{ cancelled: true }>>;
  createRefund(input): Promise<PaymentResult<RefundRecord>>;
  parseWebhook(rawBody, sig): Promise<PaymentResult<WebhookPayload>>;
}
```

`CheckoutSession` always carries `checkoutUrl: string` — the URL to redirect
users to regardless of provider.

---

## Usage

```ts
import { createPaymentProvider } from "@/lib/payment";

// Toggle with env var
const provider = createPaymentProvider(
  process.env.PAYMENT_PROVIDER === "razorpay" ? "razorpay" : "stripe",
);

// In a server action or route handler:
const result = await provider.createCheckout({
  orderId: order.id,
  amount: 4999, // paise / cents — smallest currency unit
  currency: "inr",
  customerEmail: user.email,
  successUrl: `${origin}/order/${order.id}/success`,
  cancelUrl: `${origin}/cart`,
});

if (!result.ok) {
  console.error(result.error, result.code);
  return { error: "Payment could not be initiated." };
}

redirect(result.data.checkoutUrl);
```

---

## Installation

```bash
# Stripe
npm install stripe
# Razorpay
npm install razorpay
```

---

## Environment variables

| Variable                  | Provider | Purpose                              |
| ------------------------- | -------- | ------------------------------------ |
| `STRIPE_SECRET_KEY`       | Stripe   | API auth                             |
| `STRIPE_WEBHOOK_SECRET`   | Stripe   | Webhook signature verification       |
| `RAZORPAY_KEY_ID`         | Razorpay | API key                              |
| `RAZORPAY_KEY_SECRET`     | Razorpay | API secret                           |
| `RAZORPAY_WEBHOOK_SECRET` | Razorpay | Webhook HMAC-SHA256 validation       |
| `PAYMENT_PROVIDER`        | Both     | `"stripe"` (default) or `"razorpay"` |

---

## Webhook setup

Copy the example files to `app/api/webhooks/stripe/route.ts` and
`app/api/webhooks/razorpay/route.ts`. Both use `export const dynamic = "force-dynamic"`
to ensure the raw body is not consumed by Next.js middleware.

### Stripe

Register the endpoint in the Stripe dashboard. Required events:
`payment_intent.succeeded`, `payment_intent.payment_failed`,
`customer.subscription.updated`, `customer.subscription.deleted`,
`charge.refunded`.

### Razorpay

Register in Settings → Webhooks. Required events:
`payment.captured`, `payment.failed`, `subscription.activated`,
`subscription.halted`, `subscription.cancelled`, `refund.created`.

---

## What can go wrong

| Issue                                   | Cause                                                                   | Fix                                                                                 |
| --------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `checkoutUrl` is `null`                 | Stripe returned a session without a URL (rare, usually a mode mismatch) | Ensure `mode: "payment"` is set; add a `default_payment_method` if using `Customer` |
| Signature verification fails            | Raw body was parsed as JSON before reaching the handler                 | Set `runtime = "edge"` or ensure `export const dynamic = "force-dynamic"`           |
| Razorpay `createSubscription` fails     | Plan ID doesn't exist in Razorpay dashboard                             | Create the Plan there first; pass `plan_...` as `planId`                            |
| `CARD_DECLINED` vs `INSUFFICIENT_FUNDS` | Stripe distinguishes these via `err.code`                               | Already mapped in `handleStripeError()`                                             |
| Amount off by 100×                      | Input `amount` must be in smallest unit (paise/cents)                   | Convert: `Math.round(priceInRupees * 100)`                                          |

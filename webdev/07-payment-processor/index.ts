/**
 * index.ts — barrel export + factory for the payment processor pattern.
 *
 * Usage:
 *   const provider = createPaymentProvider("stripe");
 *   const result = await provider.createCheckout({ ... });
 *   if (result.ok) console.log(result.data.checkoutUrl);
 */
export type {
  PaymentProvider,
  PaymentResult,
  PaymentErrorCode,
  CreateCheckoutInput,
  CheckoutSession,
  CreateCustomerInput,
  CustomerRecord,
  CreateSubscriptionInput,
  SubscriptionRecord,
  CreateRefundInput,
  RefundRecord,
  WebhookPayload,
} from "./types";

export { paymentOk, paymentError } from "./types";
export { StripeProvider } from "./stripe";
export { RazorpayProvider } from "./razorpay";

import { StripeProvider } from "./stripe";
import { RazorpayProvider } from "./razorpay";
import type { PaymentProvider } from "./types";

// ─── Factory ──────────────────────────────────────────────────────────────────

/**
 * Creates a PaymentProvider instance for the given provider name.
 *
 * The provider is instantiated lazily so env vars are read at call time, not
 * at module load time — which is important in edge runtimes that populate env
 * vars after module initialisation.
 *
 * @example
 *   // server action or route handler
 *   const provider = createPaymentProvider(
 *     process.env.PAYMENT_PROVIDER === "razorpay" ? "razorpay" : "stripe"
 *   );
 */
export function createPaymentProvider(
  name: "stripe" | "razorpay",
): PaymentProvider {
  switch (name) {
    case "stripe":
      return new StripeProvider();
    case "razorpay":
      return new RazorpayProvider();
    default: {
      // Exhaustiveness guard
      const _unreachable: never = name;
      throw new Error(`Unknown payment provider: ${String(_unreachable)}`);
    }
  }
}

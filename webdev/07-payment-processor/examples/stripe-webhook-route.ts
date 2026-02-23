/**
 * Stripe webhook route handler.
 * Place at: app/api/webhooks/stripe/route.ts
 *
 * Dashboard setting: Stripe → Developers → Webhooks → Add endpoint
 * Events to listen for:
 *   payment_intent.succeeded, payment_intent.payment_failed,
 *   customer.subscription.updated, customer.subscription.deleted,
 *   charge.refunded
 */
import { headers } from "next/headers";
import { NextResponse } from "next/server";
import { createPaymentProvider } from "../../index";

// Next.js must receive the raw body — disable JSON body parsing.
export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<NextResponse> {
  const sig = (await headers()).get("stripe-signature");
  if (!sig) {
    return NextResponse.json(
      { error: "Missing stripe-signature header." },
      { status: 400 },
    );
  }

  const rawBody = await req.text();
  const stripe = createPaymentProvider("stripe");
  const result = await stripe.parseWebhook(rawBody, sig);

  if (!result.ok) {
    console.error("[stripe-webhook] parse error:", result.error);
    // Return 400 for signature failures, 200 for unhandled event types
    // so Stripe doesn't retry indefinitely.
    const status = result.code === "WEBHOOK_SIGNATURE_INVALID" ? 400 : 200;
    return NextResponse.json({ error: result.error }, { status });
  }

  const payload = result.data;

  switch (payload.event) {
    case "payment.succeeded":
      // TODO: fulfil order in DB
      console.log(
        "[stripe] payment succeeded:",
        payload.paymentId,
        payload.amount,
      );
      break;

    case "payment.failed":
      // TODO: mark order as failed, notify user
      console.log(
        "[stripe] payment failed:",
        payload.paymentId,
        payload.reason,
      );
      break;

    case "subscription.activated":
      // TODO: update user subscription_status → "active"
      console.log("[stripe] subscription activated:", payload.subscriptionId);
      break;

    case "subscription.past_due":
      // TODO: update user subscription_status → "past_due", send reminder
      console.log("[stripe] subscription past due:", payload.subscriptionId);
      break;

    case "subscription.cancelled":
      // TODO: update user subscription_status → "cancelled", restrict access
      console.log("[stripe] subscription cancelled:", payload.subscriptionId);
      break;

    case "refund.created":
      // TODO: update order with refund info
      console.log("[stripe] refund created:", payload.refundId, payload.amount);
      break;
  }

  return NextResponse.json({ received: true });
}

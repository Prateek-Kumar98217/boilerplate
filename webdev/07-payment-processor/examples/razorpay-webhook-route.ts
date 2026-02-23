/**
 * Razorpay webhook route handler.
 * Place at: app/api/webhooks/razorpay/route.ts
 *
 * Dashboard setting: Razorpay → Settings → Webhooks → + Add New Webhook
 * Active events:
 *   payment.captured, payment.failed,
 *   subscription.activated, subscription.halted, subscription.cancelled,
 *   refund.created
 *
 * Razorpay sends the signature in the "x-razorpay-signature" header.
 */
import { headers } from "next/headers";
import { NextResponse } from "next/server";
import { createPaymentProvider } from "../../index";

export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<NextResponse> {
  const sig = (await headers()).get("x-razorpay-signature");
  if (!sig) {
    return NextResponse.json(
      { error: "Missing x-razorpay-signature header." },
      { status: 400 },
    );
  }

  const rawBody = await req.text();
  const razorpay = createPaymentProvider("razorpay");
  const result = await razorpay.parseWebhook(rawBody, sig);

  if (!result.ok) {
    console.error("[razorpay-webhook] parse error:", result.error);
    const status = result.code === "WEBHOOK_SIGNATURE_INVALID" ? 400 : 200;
    return NextResponse.json({ error: result.error }, { status });
  }

  const payload = result.data;

  switch (payload.event) {
    case "payment.succeeded":
      // TODO: fulfil order in DB
      console.log(
        "[razorpay] payment captured:",
        payload.paymentId,
        payload.amount,
      );
      break;

    case "payment.failed":
      // TODO: mark order as failed
      console.log(
        "[razorpay] payment failed:",
        payload.paymentId,
        payload.reason,
      );
      break;

    case "subscription.activated":
      // TODO: update user subscription_status → "active"
      console.log("[razorpay] subscription activated:", payload.subscriptionId);
      break;

    case "subscription.past_due":
      // TODO: update user subscription_status → "past_due"
      console.log(
        "[razorpay] subscription halted/past_due:",
        payload.subscriptionId,
      );
      break;

    case "subscription.cancelled":
      // TODO: update user subscription_status → "cancelled"
      console.log("[razorpay] subscription cancelled:", payload.subscriptionId);
      break;

    case "refund.created":
      // TODO: record refund
      console.log(
        "[razorpay] refund created:",
        payload.refundId,
        payload.amount,
      );
      break;
  }

  // Razorpay expects a 200 response.
  return NextResponse.json({ status: "ok" });
}

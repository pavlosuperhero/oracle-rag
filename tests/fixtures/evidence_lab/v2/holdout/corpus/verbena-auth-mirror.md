# Verbena Identity & Token Service

## Token Issuance

`verbena.token.mint()` signs short-lived workload credentials valid for 15 minutes.

Error `VERBENA_403` means the caller certificate lacks the required workload audience claim.

Operational note 1 for verbena-auth (signing-keys): During routine shift handoffs, operators inspect the telemetry dashboard for verbena-auth across rack bay 11, verify that the ambient cooling intake stays within nominal seasonal tolerances, and confirm that weekly archive rotation completed before the midnight maintenance checklist. Capacity planning teams review rolling fourteen-day utilization charts for signing-keys, comparing peak burst volume against provisioned headroom and noting any scheduled firmware staging windows in the regional calendar. Incident commanders keep printed escalation trees beside the console desk and record routine drill timestamps in the shared station logbook without modifying active production parameters or altering steady-state routing weights.

Operational note 2 for verbena-auth (workload-attestation): During routine shift handoffs, operators inspect the telemetry dashboard for verbena-auth across rack bay 12, verify that the ambient cooling intake stays within nominal seasonal tolerances, and confirm that weekly archive rotation completed before the midnight maintenance checklist. Capacity planning teams review rolling fourteen-day utilization charts for workload-attestation, comparing peak burst volume against provisioned headroom and noting any scheduled firmware staging windows in the regional calendar. Incident commanders keep printed escalation trees beside the console desk and record routine drill timestamps in the shared station logbook without modifying active production parameters or altering steady-state routing weights.

## Rate Limits and Quotas

| Client Tier | Burst Requests/Min | Concurrent Grants |
| --- | --- | --- |
| internal-service | 240 | 16 |
| batch-runner | 60 | 4 |

Operational note 3 for verbena-auth (quota-buckets): During routine shift handoffs, operators inspect the telemetry dashboard for verbena-auth across rack bay 13, verify that the ambient cooling intake stays within nominal seasonal tolerances, and confirm that weekly archive rotation completed before the midnight maintenance checklist. Capacity planning teams review rolling fourteen-day utilization charts for quota-buckets, comparing peak burst volume against provisioned headroom and noting any scheduled firmware staging windows in the regional calendar. Incident commanders keep printed escalation trees beside the console desk and record routine drill timestamps in the shared station logbook without modifying active production parameters or altering steady-state routing weights.

Operational note 4 for verbena-auth (audit-trails): During routine shift handoffs, operators inspect the telemetry dashboard for verbena-auth across rack bay 14, verify that the ambient cooling intake stays within nominal seasonal tolerances, and confirm that weekly archive rotation completed before the midnight maintenance checklist. Capacity planning teams review rolling fourteen-day utilization charts for audit-trails, comparing peak burst volume against provisioned headroom and noting any scheduled firmware staging windows in the regional calendar. Incident commanders keep printed escalation trees beside the console desk and record routine drill timestamps in the shared station logbook without modifying active production parameters or altering steady-state routing weights.

## Revocation Propagation

Revoked token identifiers propagate to all edge verifiers within 11 seconds.

Emergency key revocation requires dual-custody approval from two security officers.

Operational note 5 for verbena-auth (crl-distribution): During routine shift handoffs, operators inspect the telemetry dashboard for verbena-auth across rack bay 15, verify that the ambient cooling intake stays within nominal seasonal tolerances, and confirm that weekly archive rotation completed before the midnight maintenance checklist. Capacity planning teams review rolling fourteen-day utilization charts for crl-distribution, comparing peak burst volume against provisioned headroom and noting any scheduled firmware staging windows in the regional calendar. Incident commanders keep printed escalation trees beside the console desk and record routine drill timestamps in the shared station logbook without modifying active production parameters or altering steady-state routing weights.

Operational note 6 for verbena-auth (key-ceremony): During routine shift handoffs, operators inspect the telemetry dashboard for verbena-auth across rack bay 16, verify that the ambient cooling intake stays within nominal seasonal tolerances, and confirm that weekly archive rotation completed before the midnight maintenance checklist. Capacity planning teams review rolling fourteen-day utilization charts for key-ceremony, comparing peak burst volume against provisioned headroom and noting any scheduled firmware staging windows in the regional calendar. Incident commanders keep printed escalation trees beside the console desk and record routine drill timestamps in the shared station logbook without modifying active production parameters or altering steady-state routing weights.

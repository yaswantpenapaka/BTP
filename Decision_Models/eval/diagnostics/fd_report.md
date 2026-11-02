# Fast Decisions diagnosis (offline, saved rows only)

Rows: 255 items, 725 questions. Rows with error: {'jev': 0, 'luna': 0}. Questions with no choice: {'jev': 0, 'luna': 0}.
Recomputed item-exact-match equals saved `correct` for: {'jev': 255, 'luna': 255} of 255 items.

## 1. Accuracy: item vs head vs question
| model | ITEM exact (all heads) | HEAD exact (card-style) | per-question | q: single-label | q: multi-label per-label yes/no | heads: single | heads: multi-label (set exact) |
|---|---|---|---|---|---|---|---|
| jev | 42.7% (109/255) | 57.1% (250/438) | 69.8% (506/725) | 60.3% (n=393) | 81.0% (n=332) | 60.3% (n=393) | 28.9% (n=45) |
| luna | 42.4% (108/255) | 56.8% (249/438) | 68.6% (497/725) | 60.3% (n=393) | 78.3% (n=332) | 60.3% (n=393) | 26.7% (n=45) |

Items: both right 94, neither right 132, jev-only 15, luna-only 14

### Per domain (item exact / head exact / per-question)
| domain | items | questions | heads | jev item | jev head | jev q | luna item | luna head | luna q |
|---|---|---|---|---|---|---|---|---|---|
| agent_handoff | 14 | 14 | 1 | 78.6% | 78.6% | 78.6% | 64.3% | 64.3% | 64.3% |
| banking_intent | 16 | 16 | 1 | 43.8% | 43.8% | 43.8% | 37.5% | 37.5% | 37.5% |
| benefits_request | 17 | 34 | 2 | 52.9% | 76.5% | 76.5% | 52.9% | 73.5% | 73.5% |
| clinic_request | 16 | 32 | 2 | 50.0% | 65.6% | 65.6% | 43.8% | 65.6% | 65.6% |
| document_type | 16 | 16 | 1 | 81.2% | 81.2% | 81.2% | 87.5% | 87.5% | 87.5% |
| email_triage | 15 | 60 | 4 | 0.0% | 43.3% | 43.3% | 13.3% | 53.3% | 53.3% |
| news_topic | 15 | 15 | 1 | 80.0% | 80.0% | 80.0% | 73.3% | 73.3% | 73.3% |
| paper_field | 14 | 14 | 1 | 28.6% | 28.6% | 28.6% | 28.6% | 28.6% | 28.6% |
| product_feedback | 15 | 135 | 2 | 20.0% | 46.7% | 75.6% | 33.3% | 53.3% | 75.6% |
| restaurant_review | 14 | 98 | 2 | 14.3% | 50.0% | 78.6% | 21.4% | 53.6% | 76.5% |
| review_sentiment | 14 | 14 | 1 | 71.4% | 71.4% | 71.4% | 64.3% | 64.3% | 64.3% |
| screen_tags | 16 | 144 | 2 | 25.0% | 56.2% | 85.4% | 6.2% | 43.8% | 80.6% |
| sports_recap | 16 | 48 | 3 | 43.8% | 77.1% | 77.1% | 50.0% | 79.2% | 79.2% |
| support_intent | 14 | 14 | 1 | 42.9% | 42.9% | 42.9% | 57.1% | 57.1% | 57.1% |
| support_topic | 15 | 15 | 1 | 33.3% | 33.3% | 33.3% | 26.7% | 26.7% | 26.7% |
| ticket_route | 14 | 42 | 3 | 0.0% | 42.9% | 42.9% | 0.0% | 35.7% | 35.7% |
| travel_request | 14 | 14 | 1 | 57.1% | 57.1% | 57.1% | 57.1% | 57.1% | 57.1% |

### Per head (task), with majority-class baseline on the same test items
| domain | head | kind | #labels | n | majority baseline | jev | luna |
|---|---|---|---|---|---|---|---|
| agent_handoff | should_handoff | single | 2 | 14 | 57% (yes) | 78.6% | 64.3% |
| banking_intent | intent | single | 16 | 16 | 12% (other) | 43.8% | 37.5% |
| benefits_request | asking_status | single | 2 | 17 | 59% (yes) | 70.6% | 58.8% |
| benefits_request | program | single | 9 | 17 | 18% (pension) | 82.4% | 88.2% |
| clinic_request | request | single | 11 | 16 | 12% (records) | 56.2% | 62.5% |
| clinic_request | urgent | single | 2 | 16 | 62% (no) | 75.0% | 68.8% |
| document_type | doc_type | single | 12 | 16 | 12% (sports_recap) | 81.2% | 87.5% |
| email_triage | action | single | 4 | 15 | 40% (fyi) | 40.0% | 60.0% |
| email_triage | category | single | 8 | 15 | 20% (legal) | 26.7% | 33.3% |
| email_triage | is_phishing | single | 2 | 15 | 53% (no) | 53.3% | 60.0% |
| email_triage | needs_reply | single | 2 | 15 | 60% (no) | 53.3% | 60.0% |
| news_topic | topic | single | 10 | 15 | 13% (culture) | 80.0% | 73.3% |
| paper_field | field | single | 10 | 14 | 14% (computer_science) | 28.6% | 28.6% |
| product_feedback | feedback_type | single | 4 | 15 | 33% (bug) | 60.0% | 73.3% |
| product_feedback | product_area | multi | 8 | 15 | 13% (integrations) | 33.3% | 33.3% |
| restaurant_review | aspects | multi | 6 | 14 | 29% (ambiance/cleanliness/food/price/service/wait) | 28.6% | 35.7% |
| restaurant_review | sentiment | single | 3 | 14 | 36% (neutral) | 71.4% | 71.4% |
| review_sentiment | sentiment | single | 3 | 14 | 36% (negative) | 71.4% | 64.3% |
| screen_tags | format | single | 2 | 16 | 75% (series) | 87.5% | 75.0% |
| screen_tags | genres | multi | 8 | 16 | 12% (documentary/drama) | 25.0% | 12.5% |
| sports_recap | result | single | 4 | 16 | 44% (loss) | 93.8% | 100.0% |
| sports_recap | sport | single | 8 | 16 | 19% (cricket) | 68.8% | 68.8% |
| sports_recap | upset | single | 2 | 16 | 56% (no) | 68.8% | 68.8% |
| support_intent | intent | single | 28 | 14 | 7% (refund_request) | 42.9% | 57.1% |
| support_topic | topic | single | 12 | 15 | 13% (technical_outage) | 33.3% | 26.7% |
| ticket_route | contains_pii | single | 2 | 14 | 57% (no) | 71.4% | 57.1% |
| ticket_route | queue | single | 16 | 14 | 14% (compliance) | 28.6% | 28.6% |
| ticket_route | urgency | single | 4 | 14 | 36% (low) | 28.6% | 21.4% |
| travel_request | intent | single | 12 | 14 | 14% (baggage) | 57.1% | 57.1% |

Item-level exact match of an always-majority predictor (oracle priors from test gold): 12.9%

## 2. Questions / heads per item and the all-heads penalty
Questions per item distribution: {1: 132, 2: 33, 3: 30, 4: 15, 7: 14, 9: 31}
Heads per item distribution: {1: 132, 2: 78, 3: 30, 4: 15}

| model | actual item exact | expected if heads independent (product of per-head acc) | expected if questions independent (per-(domain,head) q acc) | global q-acc ^ #q |
|---|---|---|---|---|
| jev | 42.7% | 43.3% | 42.6% | 48.7% |
| luna | 42.4% | 41.8% | 40.6% | 47.5% |

| #heads | items | jev item | luna item | jev per-head | luna per-head |
|---|---|---|---|---|---|
| 1 head(s) | 132 | 57.6% | 55.3% | 57.6% | 55.3% |
| 2 head(s) | 78 | 33.3% | 32.1% | 59.6% | 58.3% |
| 3 head(s) | 30 | 23.3% | 26.7% | 61.1% | 58.9% |
| 4 head(s) | 15 | 0.0% | 13.3% | 43.3% | 53.3% |

| #questions | items | jev item | luna item |
|---|---|---|---|
| 1 q | 132 | 57.6% | 55.3% |
| 2-3 q | 63 | 38.1% | 38.1% |
| 4-8 q | 29 | 6.9% | 17.2% |
| 9-20 q | 31 | 22.6% | 19.4% |

## 3. Multi-label (per-label yes/no)
Items containing a multi-label head: 45; multi-label heads: 45; per-label yes/no questions: 332
| model | head | n | TP | FP | FN | TN | precision | recall | F1 | gold=yes rate | pred=yes rate | q acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | ALL | 332 | 118 | 52 | 11 | 151 | 0.694 | 0.915 | 0.789 | 38.9% | 51.2% | 81.0% |
| jev | product_area | 120 | 41 | 25 | 2 | 52 | 0.621 | 0.953 | 0.752 | 35.8% | 55.0% | 77.5% |
| jev | aspects | 84 | 51 | 15 | 2 | 16 | 0.773 | 0.962 | 0.857 | 63.1% | 78.6% | 79.8% |
| jev | genres | 128 | 26 | 12 | 7 | 83 | 0.684 | 0.788 | 0.732 | 25.8% | 29.7% | 85.2% |
| luna | ALL | 332 | 117 | 60 | 12 | 143 | 0.661 | 0.907 | 0.765 | 38.9% | 53.3% | 78.3% |
| luna | product_area | 120 | 40 | 26 | 3 | 51 | 0.606 | 0.930 | 0.734 | 35.8% | 55.0% | 75.8% |
| luna | aspects | 84 | 52 | 18 | 1 | 13 | 0.743 | 0.981 | 0.846 | 63.1% | 83.3% | 77.4% |
| luna | genres | 128 | 25 | 16 | 8 | 79 | 0.610 | 0.758 | 0.676 | 25.8% | 32.0% | 81.2% |

Set comparison per multi-label head (predicted label set vs gold set):
| model | head | n heads | set equal | differs by 1 | by 2 | by >=3 | empty pred | mean Jaccard | sizes |
|---|---|---|---|---|---|---|---|---|---|
| jev | ALL | 45 | 28.9% | 31.1% (+1 extra 9, 1 missing 5) | 20.0% | 20.0% | 0.0% | 0.645 | mean gold size 2.87 / pred size 3.78 |
| jev | product_area | 15 | 33.3% | 6.7% (+1 extra 1, 1 missing 0) | 26.7% | 33.3% | 0.0% | 0.602 | mean gold size 2.87 / pred size 4.40 |
| jev | aspects | 14 | 28.6% | 50.0% (+1 extra 5, 1 missing 2) | 0.0% | 21.4% | 0.0% | 0.714 | mean gold size 3.79 / pred size 4.71 |
| jev | genres | 16 | 25.0% | 37.5% (+1 extra 3, 1 missing 3) | 31.2% | 6.2% | 0.0% | 0.625 | mean gold size 2.06 / pred size 2.38 |
| luna | ALL | 45 | 26.7% | 26.7% (+1 extra 9, 1 missing 3) | 20.0% | 26.7% | 0.0% | 0.603 | mean gold size 2.87 / pred size 3.93 |
| luna | product_area | 15 | 33.3% | 13.3% (+1 extra 2, 1 missing 0) | 6.7% | 46.7% | 0.0% | 0.589 | mean gold size 2.87 / pred size 4.40 |
| luna | aspects | 14 | 35.7% | 28.6% (+1 extra 3, 1 missing 1) | 14.3% | 21.4% | 0.0% | 0.700 | mean gold size 3.79 / pred size 5.00 |
| luna | genres | 16 | 12.5% | 37.5% (+1 extra 4, 1 missing 2) | 37.5% | 12.5% | 0.0% | 0.531 | mean gold size 2.06 / pred size 2.56 |

Gold set size distribution (multi-label heads): {1: 11, 2: 12, 3: 9, 5: 7, 4: 2, 6: 4}

Threshold sweep on P(yes) for multi-label head exact match (offline; probs saved). 0.5 = what was scored. 'top1+' = labels with p>=0.5, or the argmax label if none:
| model | threshold | multi-label head exact |
|---|---|---|
| jev | 0.3 | 24.4% |
| jev | 0.4 | 22.2% |
| jev | 0.5 | 26.7% |
| jev | 0.6 | 28.9% |
| jev | 0.7 | 28.9% |
| jev | 0.8 | 28.9% |
| jev | top1+ | 26.7% |
| luna | 0.3 | 24.4% |
| luna | 0.4 | 26.7% |
| luna | 0.5 | 26.7% |
| luna | 0.6 | 28.9% |
| luna | 0.7 | 31.1% |
| luna | 0.8 | 26.7% |
| luna | top1+ | 26.7% |

- jev: item exact with top1+ rule for multi-label heads: 42.4% (was 42.7%)
- luna: item exact with top1+ rule for multi-label heads: 42.4% (was 42.4%)

## 4. Single-label heads: where they fail
Worst single-label heads by mean accuracy of the two models:
| domain | head | n | jev | luna |
|---|---|---|---|---|
| ticket_route | urgency | 14 | 29% | 21% |
| paper_field | field | 14 | 29% | 29% |
| ticket_route | queue | 14 | 29% | 29% |
| email_triage | category | 15 | 27% | 33% |
| support_topic | topic | 15 | 33% | 27% |
| banking_intent | intent | 16 | 44% | 38% |
| email_triage | action | 15 | 40% | 60% |
| support_intent | intent | 14 | 43% | 57% |
| email_triage | is_phishing | 15 | 53% | 60% |
| email_triage | needs_reply | 15 | 53% | 60% |

Single-label question summary:
- 393 single-label questions; jev wrong 156, luna wrong 156, both wrong 129, of which SAME wrong label 114 (88.4%), different wrong labels 15
- both wrong with the same label AND both confidence >= 0.8 (gold-noise / ambiguity suspects): 61

**ticket_route.urgency** top errors (gold -> pred: count): jev: low->high:3, normal->high:3, critical->high:2, critical->normal:1, low->normal:1; luna: low->high:3, critical->high:3, normal->high:3, critical->normal:1, low->normal:1
**paper_field.field** top errors (gold -> pred: count): jev: mathematics->computer_science:1, engineering->medicine:1, earth_science->medicine:1, computer_science->engineering:1, other->computer_science:1; luna: computer_science->medicine:1, engineering->medicine:1, earth_science->engineering:1, computer_science->engineering:1, other->mathematics:1
**ticket_route.queue** top errors (gold -> pred: count): jev: outages->payments:1, identity->payments:1, compliance->fraud_review:1, fraud_review->shipping:1, shipping->returns:1; luna: outages->payments:1, identity->billing_disputes:1, fraud_review->shipping:1, shipping->returns:1, general->payments:1
**email_triage.category** top errors (gold -> pred: count): jev: newsletter->billing:2, legal->calendar:1, newsletter->security:1, calendar->support:1, recruiting->security:1; luna: newsletter->billing:2, legal->calendar:1, billing->security:1, newsletter->legal:1, calendar->support:1
**support_topic.topic** top errors (gold -> pred: count): jev: onboarding->integrations:1, compliance->returns:1, data_privacy->account_access:1, integrations->returns:1, technical_outage->billing:1; luna: onboarding->integrations:1, compliance->returns:1, data_privacy->account_access:1, integrations->returns:1, technical_outage->billing:1
**banking_intent.intent** top errors (gold -> pred: count): jev: other->dispute_charge:1, fee_refund->direct_deposit:1, other->direct_deposit:1, statement->balance:1, statement->wire:1; luna: other->dispute_charge:1, fee_refund->direct_deposit:1, other->direct_deposit:1, statement->dispute_charge:1, statement->wire:1

### 15 misclassified examples from the worst single-label heads
1. [both wrong, SAME] `fastdecisions_banking_intent_0003` head `intent`
   - input: [channel] phone [utterance] Since moving our small-business office to Brookmere, I have been reviewing the checking account and noticed a recurring scheduled card purchase that has settled every Friday for the past several weeks. The merchant descriptor is "CivicDesk Pro," transaction code SD-7314, and the charge is $42.80 on our business debit card ending 9066. I do not recogn...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **other** | Jev: dispute_charge (dispute_charge=0.61, fraud_alert=0.34, other=0.05) | Luna: dispute_charge (dispute_charge=0.87, other=0.10, fraud_alert=0.01)
2. [both wrong, SAME] `fastdecisions_banking_intent_0016` head `intent`
   - input: [channel] branch [utterance] I’m currently living away from home for my studies and noticed that my standing order for payroll direct deposit to my personal checking hasn’t posted twice this month, though the authorization holds appeared immediately. After a few calls, I’m frustrated this duplicate transaction issue isn’t resolved yet. Could you urgently review why these paymen...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **fee_refund** | Jev: direct_deposit (direct_deposit=1.00, balance=0.00, transfer=0.00) | Luna: direct_deposit (direct_deposit=0.59, fee_refund=0.19, dispute_charge=0.10)
3. [both wrong, SAME] `fastdecisions_banking_intent_0020` head `intent`
   - input: [channel] app [utterance] I just saw a deposit that seems to be my payroll from our company but it was reversed for some reason. Since this is for our joint household account, I’m concerned this might be an error or even some unauthorized action. It’s important to get this sorted before my upcoming deadline with the supplier. Can you help check why the deposit was pulled back a...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **other** | Jev: direct_deposit (direct_deposit=0.98, dispute_charge=0.01, fraud_alert=0.01) | Luna: direct_deposit (direct_deposit=0.98, fraud_alert=0.02, balance=0.00)
4. [both wrong, SAME] `fastdecisions_banking_intent_0036` head `intent`
   - input: [channel] branch [utterance] For nearly three weeks, an authorized international wire from our joint household account has remained pending while I study away from home. The recipient’s bank says an intermediary bank may be holding it, and I need this resolved immediately. Please investigate the transfer status and explain any correspondent-bank delay. I understand APR and loan...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **statement** | Jev: wire (wire=0.99, transfer=0.01, balance=0.00) | Luna: wire (wire=0.98, transfer=0.01, branch_appointment=0.01)
5. [both wrong, SAME] `fastdecisions_banking_intent_0060` head `intent`
   - input: [channel] branch [utterance] I just replaced my lost phone and reinstalled the bank app, but I see two identical charges for my mortgage debit card from yesterday that shouldn't both have processed. My available funds show less than expected due to pending holds. I've called twice with no resolution and need this fixed immediately.
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **card_lost** | Jev: dispute_charge (dispute_charge=0.99, mortgage=0.01, balance=0.00) | Luna: dispute_charge (dispute_charge=0.86, branch_appointment=0.07, card_lost=0.02)
6. [both wrong, SAME] `fastdecisions_banking_intent_0076` head `intent`
   - input: [channel] chat [utterance] I've been running my family-owned bakery for months now, and despite making ATM cash withdrawals authorized to cover daily expenses, the posted balance on my mortgage account hasn't adjusted properly. This pending loan payment issue has persisted through several weekends, likely delayed due to holiday processing. Although my available balance reflects...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **loan_payment** | Jev: mortgage (mortgage=0.67, loan_payment=0.31, balance=0.02) | Luna: mortgage (mortgage=0.53, loan_payment=0.31, other=0.04)
7. [both wrong, SAME] `fastdecisions_banking_intent_0087` head `intent`
   - input: [channel] chat [utterance] I need the duplicated ACH payment on our small-business account corrected today. While I have been studying away from home, our monthly housing payment was pulled twice from the business checking account, even though only one withdrawal was scheduled. The extra debit is putting a hold on funds needed for regular expenses, and I expect an immediate rev...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **fee_refund** | Jev: dispute_charge (dispute_charge=0.79, mortgage=0.16, loan_payment=0.03) | Luna: dispute_charge (dispute_charge=0.82, mortgage=0.14, loan_payment=0.02)
8. [both wrong, DIFFERENT] `fastdecisions_banking_intent_0026` head `intent`
   - input: [channel] app [utterance] Since joining Breakwater Consulting, I've had recurring discrepancies between the posted and available balances on my credit card account following cash remittances at the ATM. A withdrawal processed three weeks ago was reversed, yet the charge continues to reflect in my monthly statement. Could you clarify whether this issue requires additional busine...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **statement** | Jev: balance (balance=0.91, dispute_charge=0.08, statement=0.01) | Luna: dispute_charge (dispute_charge=0.45, balance=0.29, statement=0.09)
9. [both wrong, DIFFERENT] `fastdecisions_banking_intent_0065` head `intent`
   - input: [channel] branch [utterance] My new company payroll deposit was rejected when processing the ACH transfer to my small-business account, although the available and posted balances were sufficient. The deadline to correct this is in three days, and I’m unsure why the transaction was declined—can someone help clarify the reason and steps to resolve this promptly?
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **wire** | Jev: transfer (transfer=0.63, direct_deposit=0.37, balance=0.00) | Luna: direct_deposit (direct_deposit=0.98, transfer=0.01, branch_appointment=0.01)
10. [both wrong, DIFFERENT] `fastdecisions_email_triage_0025` head `action`
   - input: From: Fiona Elridge <fiona.elridge@hightower-news.example> To: ops-team@hightower-news.example Subject: Clarification on recurring subscription billing for monthly newsletter  Dear Operations,  Following up on your question regarding the monthly invoicing cycle for our newsletter subscription: our billing department confirms that the invoicing is scheduled to recur on the first...
   - candidates: reply, approve, fyi, escalate
   - gold: **escalate** | Jev: reply (reply=0.89, fyi=0.11, approve=0.00) | Luna: fyi (fyi=0.75, reply=0.24, approve=0.01)
11. [both wrong, DIFFERENT] `fastdecisions_email_triage_0044` head `category`
   - input: From: Alex Mercer <alex.mercer@externalcounsel.lawfirm> To: ops.lead@cityoffice.gov.uk Subject: FW: Monthly data submission schedule update  Dear Ops Lead,  Apologies for any disruption, but the attached portal link reflects a modification to the monthly compliance report submissions originally scheduled every first Monday. Kindly verify this via the portal's secure token proce...
   - candidates: billing, support, sales, legal, recruiting, calendar, newsletter, security
   - gold: **newsletter** | Jev: security (security=0.41, legal=0.29, calendar=0.28) | Luna: legal (legal=0.51, security=0.35, support=0.08)
12. [only Jev wrong] `fastdecisions_email_triage_0023` head `action`
   - input: From: Adrian Cross <adrian.cross@seabreeze-air.example> To: security@seabreeze-air.example Subject: Follow-up on ticket #4852 regarding access credential reset  Despite the earlier notice, I need an update on the access credentials reset that was requested four days ago under ticket #4852. The alert system flagged unusual login attempts yesterday, which raised concerns about po...
   - candidates: reply, approve, fyi, escalate
   - gold: **reply** | Jev: escalate (escalate=0.61, reply=0.39, approve=0.00) | Luna: reply (reply=0.74, escalate=0.24, fyi=0.02)
13. [only Jev wrong] `fastdecisions_email_triage_0046` head `action`
   - input: From: Elena Maric <elena.maric@civic-access.example> To: permits@harbourview-city.example Subject: Clarification: QBR access review meeting on 22 October, CET  Dear Permits Team,  Apologies for the inconvenience caused by my earlier message. You asked whether the access review meeting remains scheduled; it is confirmed for 22 October at 14:00 CET. The calendar invitation has be...
   - candidates: reply, approve, fyi, escalate
   - gold: **fyi** | Jev: reply (reply=0.90, fyi=0.10, approve=0.00) | Luna: fyi (fyi=0.69, reply=0.29, approve=0.01)
14. [only Luna wrong] `fastdecisions_banking_intent_0044` head `intent`
   - input: [channel] branch [utterance] We've been managing our joint household mortgage account for some years, and after reviewing our recent escrow analysis, it appears a monthly online bill payment reflected twice, but one payment has since been reversed. Since I'm currently traveling abroad and concerned about potential unauthorized transactions, could you help me confirm the account...
   - candidates: balance, transfer, card_lost, dispute_charge, loan_payment, mortgage, wire, statement, fraud_alert, branch_appointment, fee_refund, direct_deposit, close_account, new_account, limit_increase, other
   - gold: **close_account** | Jev: close_account (close_account=0.78, mortgage=0.22, balance=0.00) | Luna: mortgage (mortgage=0.76, close_account=0.17, fraud_alert=0.05)
15. [only Luna wrong] `fastdecisions_email_triage_0038` head `category`
   - input: From: Kieran Walsh <kieran.walsh@toronto-city.example> To: billing@toronto-city.example Subject: Forwarded: clarification regarding invoice #4523  Team,  I'm forwarding this message I received from our water services provider. The invoice includes a new payment method, and they're requesting the balance to be settled within two business days. I haven’t authorized changes to pay...
   - candidates: billing, support, sales, legal, recruiting, calendar, newsletter, security
   - gold: **billing** | Jev: billing (billing=0.54, security=0.46, support=0.00) | Luna: security (security=0.70, billing=0.27, support=0.02)

## 5. Adapter vs dataset card
- test rows vs original raw file line (item_id index): 0 mismatches of 255.
- data integrity: heads=438, gold_not_in_labels=0, dup_candidate_labels=0, labels_not_stripped_or_nonlower=0, single_with_len_ne_1=0, multi_with_0_gold=0, multi_with_1_gold=11, multi_with_2plus_gold=34, gold_listed_duplicates=0, heads_in_row_same_task_name=0, items_with_duplicate_task_names=0
- adapter options/labels differ from card labels (order or content): 0; adapter gold != card true_label: 0
- Option keys: single-label options are sent with description == the label string itself (no extra description); yes/no heads use labels 'yes'/'no' as-is.
- Rows whose chosen option is outside the option set (would be dropped): 0; rows with notes about missing/extra option keys: 0
| model | set-exact (used) | case-insensitive | normalized (strip _ - space) | ordered-list compare (pred order = candidate order) |
|---|---|---|---|---|
| jev | 42.7% | 42.7% | 42.7% | 40.8% |
| luna | 42.4% | 42.4% | 42.4% | 39.6% |

- multi-label gold lists NOT in candidate-label order (so ordered-list compare would be wrong, set compare is right): 18

## 6. Duplicates and ambiguity
- full dev set (1,700 rows): identical (normalized) inputs within a domain: 0 groups; with DIFFERENT gold: 0
- test split: duplicate inputs 0
- near-duplicate pairs (word Jaccard >= 0.8) within a domain in the 1,700 dev rows: 0; with different gold: 0
- Candidate label sets with overlapping meaning (labels sharing a word, or containing 'other'), per single-label head:
| domain | head | #labels | has 'other' | labels sharing a word |
|---|---|---|---|---|
| banking_intent | intent | 16 | yes | close_account~new_account |
| benefits_request | program | 9 | yes | - |
| clinic_request | request | 11 | yes | - |
| document_type | doc_type | 12 | no | - |
| email_triage | action | 4 | no | - |
| email_triage | category | 8 | no | - |
| news_topic | topic | 10 | no | - |
| paper_field | field | 10 | yes | computer_science~earth_science |
| product_feedback | feedback_type | 4 | no | - |
| product_feedback | product_area | 8 | no | - |
| restaurant_review | aspects | 6 | no | - |
| restaurant_review | sentiment | 3 | no | - |
| review_sentiment | sentiment | 3 | no | - |
| screen_tags | format | 2 | no | - |
| screen_tags | genres | 8 | no | - |
| sports_recap | result | 4 | no | - |
| sports_recap | sport | 8 | yes | - |
| support_intent | intent | 28 | yes | order_status~change_order, refund_request~feature_request, missing_item~damaged_item, change_address~change_order, product_question~pricing_question, upgrade_plan~downgrade_plan |
| support_topic | topic | 12 | yes | - |
| ticket_route | queue | 16 | no | - |
| ticket_route | urgency | 4 | no | - |
| travel_request | intent | 12 | yes | - |

- jev: accuracy when gold is 'other' = 0.0% (n=13); non-'other' gold in heads that have an 'other' option: predicted 'other' wrongly in 0.9% of 109
- luna: accuracy when gold is 'other' = 7.7% (n=13); non-'other' gold in heads that have an 'other' option: predicted 'other' wrongly in 1.8% of 109
- Among single-label questions both models got wrong (129), they gave the same label in 114: strongly suggests label ambiguity/gold noise on a part of the data rather than random model error. Per-domain counts of same-wrong: {'email_triage': 22, 'ticket_route': 20, 'support_topic': 9, 'sports_recap': 8, 'banking_intent': 7, 'clinic_request': 7, 'paper_field': 6, 'travel_request': 6}

## 7. Request actually sent (reconstructed with the saved adapter + client body builders; nothing is sent)
### fastdecisions_support_intent_0000 (1 questions)
```json
{"model": "jev-1.13.0", "state": {"input": "[channel] email\n[utterance] Dear support team, I reached out last week regarding issues with my monthly subscription renewal for order SUB-204859, and after fol…[+427 chars]"}, "questions": {"q0": {"type": "choice", "instructions": "Task 'intent'. Which label best fits `input`?", "criteria": {"order_status": "order_status", "refund_request": "refund_request", "cancel_subscription": "cancel_subscription", "update_payment": "update_payment", "login_problem": "login_problem", "password_reset": "password_reset", "shipping_delay": "shipping_delay", "missing_item": "missing_item", "damaged_item": "damaged_item", "change_address": "change_address", "change_order": "change_order", "product_question": "product_question", "pricing_question": "pricing_question", "upgrade_plan": "upgrade_plan", "downgrade_plan": "downgrade_plan", "bug_report": "bug_report", "feature_request": "feature_request", "speak_to_human": "speak_to_human", "complaint": "complaint", "invoice_copy": "invoice_copy", "tax_document": "tax_document", "appointment": "appointment", "outage": "outage", "data_export": "data_export", "close_account": "close_account", "promo_code": "promo_code", "partner_inquiry": "partner_inquiry", "other": "other"}}}}
```
```json
{"model": "gpt-6-luna", "input": "{\"input\": \"[channel] email\\n[utterance] Dear support team, I reached out last week regarding issues with my monthly subscription renewal for order SUB-204859, a…[+441 chars]", "questions": [{"type": "choice", "name": "q0", "instructions": "Task 'intent'. Which label best fits `input`?", "choices": [{"value": "order_status", "description": "order_status"}, {"value": "refund_request", "description": "refund_request"}, {"value": "cancel_subscription", "description": "cancel_subscription"}, {"value": "update_payment", "description": "update_payment"}, {"value": "login_problem", "description": "login_problem"}, {"value": "password_reset", "description": "password_reset"}, {"value": "shipping_delay", "description": "shipping_delay"}, {"value": "missing_item", "description": "missing_item"}, {"value": "damaged_item", "description": "damaged_item"}, {"value": "change_address", "description": "change_address"}, {"value": "change_order", "description": "change_order"}, {"value": "product_question", "description": "product_question"}, {"value": "pricing_question", "description": "pricing_question"}, {"value": "upgrade_plan", "description": "upgrade_plan"}, {"value": "downgrade_plan", "description": "downgrade_plan"}, {"value": "bug_report", "description": "bug_report"}, {"value": "feature_request", "description": "feature_request"}, {"value": "speak_to_human", "description": "speak_to_human"}, {"value": "complaint", "description": "complaint"}, {"value": "invoice_copy", "description": "invoice_copy"}, {"value": "tax_document", "description": "tax_document"}, {"value": "appointment", "description": "appointment"}, {"value": "outage", "description": "outage"}, {"value": "data_export", "description": "data_export"}, {"value": "close_account", "description": "close_account"}, {"value": "promo_code", "description": "promo_code"}, {"value": "partner_inquiry", "description": "partner_inquiry"}, {"value": "other", "description": "other"}]}]}
```
### fastdecisions_email_triage_0004 (4 questions)
```json
{"model": "jev-1.13.0", "state": {"input": "From: Amira Clarke <amira.clarke@lakesideclinic.example>\nTo: legal@lakesideclinic.example\nSubject: Clarification needed: urgent amendment to patient consent for…[+926 chars]"}, "questions": {"q0": {"type": "choice", "instructions": "Task 'category'. Which label best fits `input`?", "criteria": {"billing": "billing", "support": "support", "sales": "sales", "legal": "legal", "recruiting": "recruiting", "calendar": "calendar", "newsletter": "newsletter", "security": "security"}}, "q1": {"type": "choice", "instructions": "Task 'action'. Which label best fits `input`?", "criteria": {"reply": "reply", "approve": "approve", "fyi": "fyi", "escalate": "escalate"}}, "q2": {"type": "choice", "instructions": "Task 'needs_reply'. Which label best fits `input`?", "criteria": {"yes": "yes", "no": "no"}}, "q3": {"type": "choice", "instructions": "Task 'is_phishing'. Which label best fits `input`?", "criteria": {"yes": "yes", "no": "no"}}}}
```
```json
{"model": "gpt-6-luna", "input": "{\"input\": \"From: Amira Clarke <amira.clarke@lakesideclinic.example>\\nTo: legal@lakesideclinic.example\\nSubject: Clarification needed: urgent amendment to patien…[+955 chars]", "questions": [{"type": "choice", "name": "q0", "instructions": "Task 'category'. Which label best fits `input`?", "choices": [{"value": "billing", "description": "billing"}, {"value": "support", "description": "support"}, {"value": "sales", "description": "sales"}, {"value": "legal", "description": "legal"}, {"value": "recruiting", "description": "recruiting"}, {"value": "calendar", "description": "calendar"}, {"value": "newsletter", "description": "newsletter"}, {"value": "security", "description": "security"}]}, {"type": "choice", "name": "q1", "instructions": "Task 'action'. Which label best fits `input`?", "choices": [{"value": "reply", "description": "reply"}, {"value": "approve", "description": "approve"}, {"value": "fyi", "description": "fyi"}, {"value": "escalate", "description": "escalate"}]}, {"type": "choice", "name": "q2", "instructions": "Task 'needs_reply'. Which label best fits `input`?", "choices": [{"value": "yes", "description": "yes"}, {"value": "no", "description": "no"}]}, {"type": "choice", "name": "q3", "instructions": "Task 'is_phishing'. Which label best fits `input`?", "choices": [{"value": "yes", "description": "yes"}, {"value": "no", "description": "no"}]}]}
```
### fastdecisions_screen_tags_0006 (9 questions)
```json
{"model": "jev-1.13.0", "state": {"input": "[blurb]\nWhen lone courier Tavi Rusk reaches the mountain hamlet of Bracken Vale, the villagers bar their doors and conceal the antique radio towers that once gu…[+528 chars]"}, "questions": {"q0_0": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'comedy' apply to `input`?", "criteria": {"yes": "The label 'comedy' applies.", "no": "The label 'comedy' does not apply."}}, "q0_1": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'drama' apply to `input`?", "criteria": {"yes": "The label 'drama' applies.", "no": "The label 'drama' does not apply."}}, "q0_2": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'action' apply to `input`?", "criteria": {"yes": "The label 'action' applies.", "no": "The label 'action' does not apply."}}, "q0_3": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'documentary' apply to `input`?", "criteria": {"yes": "The label 'documentary' applies.", "no": "The label 'documentary' does not apply."}}, "q0_4": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'horror' apply to `input`?", "criteria": {"yes": "The label 'horror' applies.", "no": "The label 'horror' does not apply."}}, "q0_5": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'romance' apply to `input`?", "criteria": {"yes": "The label 'romance' applies.", "no": "The label 'romance' does not apply."}}, "q0_6": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'sci_fi' apply to `input`?", "criteria": {"yes": "The label 'sci_fi' applies.", "no": "The label 'sci_fi' does not apply."}}, "q0_7": {"type": "choice", "instructions": "Task 'genres' (multi-label). Does the label 'kids' apply to `input`?", "criteria": {"yes": "The label 'kids' applies.", "no": "The label 'kids' does not apply."}}, "q1": {"type": "choice", "instructions": "Task 'format'. Which label best fits `input`?", "criteria": {"film": "film", "series": "series"}}}}
```
(body truncated for display; 9 questions)
```json
{"model": "gpt-6-luna", "input": "{\"input\": \"[blurb]\\nWhen lone courier Tavi Rusk reaches the mountain hamlet of Bracken Vale, the villagers bar their doors and conceal the antique radio towers …[+542 chars]", "questions": [{"type": "choice", "name": "q0_0", "instructions": "Task 'genres' (multi-label). Does the label 'comedy' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'comedy' applies."}, {"value": "no", "description": "The label 'comedy' does not apply."}]}, {"type": "choice", "name": "q0_1", "instructions": "Task 'genres' (multi-label). Does the label 'drama' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'drama' applies."}, {"value": "no", "description": "The label 'drama' does not apply."}]}, {"type": "choice", "name": "q0_2", "instructions": "Task 'genres' (multi-label). Does the label 'action' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'action' applies."}, {"value": "no", "description": "The label 'action' does not apply."}]}, {"type": "choice", "name": "q0_3", "instructions": "Task 'genres' (multi-label). Does the label 'documentary' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'documentary' applies."}, {"value": "no", "description": "The label 'documentary' does not apply."}]}, {"type": "choice", "name": "q0_4", "instructions": "Task 'genres' (multi-label). Does the label 'horror' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'horror' applies."}, {"value": "no", "description": "The label 'horror' does not apply."}]}, {"type": "choice", "name": "q0_5", "instructions": "Task 'genres' (multi-label). Does the label 'romance' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'romance' applies."}, {"value": "no", "description": "The label 'romance' does not apply."}]}, {"type": "choice", "name": "q0_6", "instructions": "Task 'genres' (multi-label). Does the label 'sci_fi' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'sci_fi' applies."}, {"value": "no", "description": "The label 'sci_fi' does not apply."}]}, {"type": "choice", "name": "q0_7", "instructions": "Task 'genres' (multi-label). Does the label 'kids' apply to `input`?", "choices": [{"value": "yes", "description": "The label 'kids' applies."}, {"value": "no", "description": "The label 'kids' 
```
Mean input_tokens per item: jev 729, luna 572

## Addendum: card-comparable numbers
- jev: domain-macro HEAD exact = 57.4%, micro HEAD exact = 57.1%, domain-macro ITEM exact = 42.5%
  bootstrap 95% CI of domain-macro head exact: [53.0, 61.9]
- luna: domain-macro HEAD exact = 56.1%, micro HEAD exact = 56.8%, domain-macro ITEM exact = 42.2%
  bootstrap 95% CI of domain-macro head exact: [51.5, 60.9]

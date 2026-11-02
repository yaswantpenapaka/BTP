---
license: apache-2.0
language:
- en
task_categories:
- text-classification
pretty_name: Fast Decisions
size_categories:
- 1K<n<10K
tags:
- gliner2
- classification
- multi-task
- decision-model
- schema-driven
configs:
- config_name: support_intent
  data_files: "support_intent.jsonl"
- config_name: support_topic
  data_files: "support_topic.jsonl"
- config_name: document_type
  data_files: "document_type.jsonl"
- config_name: review_sentiment
  data_files: "review_sentiment.jsonl"
- config_name: agent_handoff
  data_files: "agent_handoff.jsonl"
- config_name: email_triage
  data_files: "email_triage.jsonl"
- config_name: ticket_route
  data_files: "ticket_route.jsonl"
- config_name: product_feedback
  data_files: "product_feedback.jsonl"
- config_name: banking_intent
  data_files: "banking_intent.jsonl"
- config_name: clinic_request
  data_files: "clinic_request.jsonl"
- config_name: travel_request
  data_files: "travel_request.jsonl"
- config_name: news_topic
  data_files: "news_topic.jsonl"
- config_name: paper_field
  data_files: "paper_field.jsonl"
- config_name: sports_recap
  data_files: "sports_recap.jsonl"
- config_name: restaurant_review
  data_files: "restaurant_review.jsonl"
- config_name: benefits_request
  data_files: "benefits_request.jsonl"
- config_name: screen_tags
  data_files: "screen_tags.jsonl"
---

<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">
  <a href="https://fastino.ai?utm_source=huggingface" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Finetune-GLiNER2.5-EA4335" alt="Fine-tune and Deploy GLiNER2.5 with Fastino Agent" style="vertical-align:middle;">
  </a>
  <a href="https://arxiv.org/abs/2507.18546" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/arXiv-2507.18546-b31b1b.svg?logo=arxiv" alt="arXiv Paper" style="vertical-align:middle;">
  </a>
  <a href="https://github.com/fastino-ai/GLiNER2" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/GitHub-GLiNER2-black?logo=github" alt="GitHub" style="vertical-align:middle;">
  </a>
  <a href="https://x.com/fastinoAI" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/twitter/follow/:fastinoAI" alt="Follow @fastinoAI" style="vertical-align:middle;">
  </a>
</div>

# Fast Decisions

**The classification suite behind GLiNER2.5-Decide.** Seventeen domains, one file each. Every row is a document plus the decisions a product has to make: the task name, the candidate labels, and the gold label. One call can carry several heads. Single-label heads have one gold string. Multi-label heads list every label that applies.

This release is the development split: 100 examples per domain, 1,700 rows. The test split, 300 per domain, is held out. The numbers below are that held-out split. Do not report a score computed on the files in this repo as the benchmark.

This suite is not a general-purpose exam. There is no reasoning trace and no open answer. It is operational decisions: customer and banking intent, travel and clinic requests, review sentiment, document type, email and ticket routing, human handoff, and ordinal or yes/no gates.

## Schema

Every row has this shape:

```json
{
  "input": "<free text>",
  "output": {
    "classifications": [
      {
        "task": "<head name>",
        "true_label": [
          "<gold label(s)>"
        ],
        "labels": [
          "<label>",
          "..."
        ],
        "multi_label": false
      }
    ]
  }
}
```

- `true_label` is always a list of strings, even for single-label heads.
- `labels` is the **full candidate set** for that head — pass it to your model as-is; do not shuffle out the gold label.
- Rows are classification only. There is no entity field.
- Ordinal heads (0–5, 0–10) use string digits, not integers.

## Domains

| File | Domain | Group | Heads | Multi-label heads | Rows |
|---|---|---|---:|---|---:|
| `support_intent.jsonl` | Commerce support intent | commerce | 1 | — | 100 |
| `support_topic.jsonl` | Commerce support topic | commerce | 1 | — | 100 |
| `document_type.jsonl` | Document type | records | 1 | — | 100 |
| `review_sentiment.jsonl` | Review sentiment | reviews | 1 | — | 100 |
| `agent_handoff.jsonl` | Assistant handoff | assistants | 1 | — | 100 |
| `email_triage.jsonl` | Email triage | workplace | 4 | — | 100 |
| `ticket_route.jsonl` | Commerce ticket route | commerce | 3 | — | 100 |
| `product_feedback.jsonl` | Product feedback | commerce | 2 | `product_area` | 100 |
| `banking_intent.jsonl` | Banking intent | finance | 1 | — | 100 |
| `clinic_request.jsonl` | Clinic request | health | 2 | — | 100 |
| `travel_request.jsonl` | Travel request | travel | 1 | — | 100 |
| `news_topic.jsonl` | News topic | news | 1 | — | 100 |
| `paper_field.jsonl` | Paper field | science | 1 | — | 100 |
| `sports_recap.jsonl` | Sports recap | sports | 3 | — | 100 |
| `restaurant_review.jsonl` | Restaurant review | food | 2 | `aspects` | 100 |
| `benefits_request.jsonl` | Benefits request | civic | 2 | — | 100 |
| `screen_tags.jsonl` | Screen tags | entertainment | 2 | `genres` | 100 |

Total: **1,700 rows** across 17 files, 100 rows each.

## Task heads and label sets

Full label inventory per head. Use these exact strings as the candidate set at inference time.

**`support_intent`** — Commerce support intent (commerce)

- `intent`: `order_status`, `refund_request`, `cancel_subscription`, `update_payment`, `login_problem`, `password_reset`, `shipping_delay`, `missing_item`, `damaged_item`, `change_address`, … (+18 more)

**`support_topic`** — Commerce support topic (commerce)

- `topic`: `billing`, `account_access`, `shipping`, `returns`, `technical_outage`, `onboarding`, `integrations`, `security`, `compliance`, `pricing`, … (+2 more)

**`document_type`** — Document type (records)

- `doc_type`: `invoice`, `receipt`, `contract`, `resume`, `bank_statement`, `clinical_note`, `syllabus`, `news_article`, `lease`, `research_abstract`, … (+2 more)

**`review_sentiment`** — Review sentiment (reviews)

- `sentiment`: `negative`, `neutral`, `positive`

**`agent_handoff`** — Assistant handoff (assistants)

- `should_handoff`: `yes`, `no`

**`email_triage`** — Email triage (workplace)

- `category`: `billing`, `support`, `sales`, `legal`, `recruiting`, `calendar`, `newsletter`, `security`
- `action`: `reply`, `approve`, `fyi`, `escalate`
- `needs_reply`: `yes`, `no`
- `is_phishing`: `yes`, `no`

**`ticket_route`** — Commerce ticket route (commerce)

- `queue`: `billing_disputes`, `refunds`, `shipping`, `returns`, `identity`, `payments`, `outages`, `mobile_app`, `api`, `account_closure`, … (+6 more)
- `urgency`: `low`, `normal`, `high`, `critical`
- `contains_pii`: `yes`, `no`

**`product_feedback`** — Product feedback (commerce)

- `feedback_type`: `bug`, `feature_request`, `how_to`, `praise`
- `product_area` *(multi-label)*: `billing`, `mobile`, `search`, `checkout`, `shipping`, `notifications`, `admin`, `integrations`

**`banking_intent`** — Banking intent (finance)

- `intent`: `balance`, `transfer`, `card_lost`, `dispute_charge`, `loan_payment`, `mortgage`, `wire`, `statement`, `fraud_alert`, `branch_appointment`, … (+6 more)

**`clinic_request`** — Clinic request (health)

- `request`: `appointment`, `refill`, `lab_results`, `referral`, `billing`, `insurance`, `records`, `symptom_question`, `cancel_visit`, `provider_message`, … (+1 more)
- `urgent`: `yes`, `no`

**`travel_request`** — Travel request (travel)

- `intent`: `new_booking`, `change_flight`, `cancel_trip`, `seat`, `baggage`, `refund`, `hotel`, `car`, `visa_docs`, `loyalty`, … (+2 more)

**`news_topic`** — News topic (news)

- `topic`: `politics`, `business`, `technology`, `science`, `health`, `sports`, `entertainment`, `world`, `culture`, `weather`

**`paper_field`** — Paper field (science)

- `field`: `biology`, `medicine`, `chemistry`, `physics`, `computer_science`, `mathematics`, `earth_science`, `engineering`, `economics`, `other`

**`sports_recap`** — Sports recap (sports)

- `sport`: `football`, `basketball`, `baseball`, `soccer`, `tennis`, `cricket`, `motorsport`, `other`
- `result`: `win`, `loss`, `draw`, `upcoming`
- `upset`: `yes`, `no`

**`restaurant_review`** — Restaurant review (food)

- `sentiment`: `negative`, `neutral`, `positive`
- `aspects` *(multi-label)*: `food`, `service`, `price`, `ambiance`, `wait`, `cleanliness`

**`benefits_request`** — Benefits request (civic)

- `program`: `unemployment`, `health_coverage`, `food_assistance`, `housing_aid`, `childcare`, `pension`, `disability`, `tax_credit`, `other`
- `asking_status`: `yes`, `no`

**`screen_tags`** — Screen tags (entertainment)

- `genres` *(multi-label)*: `comedy`, `drama`, `action`, `documentary`, `horror`, `romance`, `sci_fi`, `kids`
- `format`: `film`, `series`

## Examples

One worked example per domain (truncated for readability where long).

### Commerce support intent (`support_intent`)

Route a live message before a human sees it. Refunds, cancellations, and shipping delays share one inbox and one label set.

```text
[channel] email
[utterance] Dear support team, I reached out last week regarding issues with my monthly subscription renewal for order SUB-204859, and after following the suggested troubleshooting steps, the problem seems unresolved. The latest invoice dated 2024-04-15 still shows a charge of ¥5,400 including consumption tax, which I did not authorize due to the service interruption. Before I request a refund, could …
```

Gold labels:

- **intent** → `refund_request`

### Commerce support topic (`support_topic`)

The same inbox, a different question: which product area owns the failure.

```text
[subject] Product page sign-in error during renewal review
[body] During our latest review, Meera at Kaveri Homeware found the product browsing screen failing before any order was placed. The merchant account showed an account-lockout message, although the failed sign-in flow began after upgrading the Checkout SDK from v4.8 to v4.9; /catalog/session and /customer/token now return 502. Subscription renewal for the …
```

Gold labels:

- **topic** → `technical_outage`

### Document type (`document_type`)

Inboxes mix invoices, contracts, and resumes. The type is the gate in front of extraction.

```text
[document]
ARCHIVED CAREER PROFILE — RIVERGLEN COMMUNITY LEARNING HUB
Record ref: Q7N-204 | Filed: 14 Aug 2023 | Prepared by: L. Maren, program coordinator

For the instructor’s review, this historical profile summarizes Elena Voss’s service as a literacy tutor and workshop assistant. She planned beginner reading sessions, explained assignments in clear language, and kept attendance totals for evening groups. From …
```

Gold labels:

- **doc_type** → `resume`

### Review sentiment (`review_sentiment`)

A product can be praised and rejected in one paragraph. The label is what a reply policy reads.

```text
[course] We just finished our second run of the Little Explorers Science Lab with our kids after loving the trial class last season. Was the menu of hands-on demos and take-home snack kits as good as advertised? Not really. Food quality: The promised "healthy fruit platters" were mostly store-bought granola bars and juice boxes this time, so that was a letdown. Service at the tables: The helpers answered questions …
```

Gold labels:

- **sentiment** → `neutral`

### Assistant handoff (`agent_handoff`)

Most turns stay automated. A repeated complaint or a request the policy forbids should leave the flow.

```text
[policy] The Meadowbrook Clinic appointment assistant may assist with booking, changing, or confirming routine consultations and provide documents about standard pre-visit requirements. It must escalate requests involving medical emergencies, medication refills, billing corrections, or interactions that require a licensed clinician or office manager.
[user] I've got a yearly checkup scheduled at the Parkside branch …
```

Gold labels:

- **should_handoff** → `no`

### Email triage (`email_triage`)

One message, four decisions: what it is, what to do, whether anyone must reply, and whether it is a lure.

```text
From: Lydia Harper <lydia.harper@cogentware.example>
To: admin@impulselectronics.example
Subject: Coordination—rescheduling quarterly review deadline

Hello Admin Team,

Summary: The project review that was previously set for today must be moved to accommodate a late-breaking platform update. We request confirmation for a new time slot before 3 p.m. Eastern today so stakeholders can be notified on schedule. …
```

Gold labels:

- **category** → `calendar`
- **action** → `escalate`
- **needs_reply** → `yes`
- **is_phishing** → `no`

### Commerce ticket route (`ticket_route`)

The customer describes a problem, not a department. Queue, urgency, and whether the text contains personal data are scored together.

```text
[channel] chat
[subject] Follow-up on compliance check for delayed invoice payment
[body] After the latest order update, I noticed that the payment for invoice number INV-209384 issued on 2027-03-15 at 13:45 UTC has not been processed due to account verification issues. Despite submitting all required documentation previously, the system still restricts order modifications and payment submissions. The message …
```

Gold labels:

- **queue** → `compliance`
- **urgency** → `low`
- **contains_pii** → `yes`

### Product feedback (`product_feedback`)

Bug, feature, or praise, plus the product areas that apply. `product_area` is multi-label.

```text
[source] app review
[text] I was trying to finalize my purchase during the holiday rush using my tablet’s browser, but the notifications about my multiple shipment orders kept repeating and were out of sync with the payment authorization updates. It made tracking which items had their subscriptions renewed a mess. Each notification seemed to loop, forcing me to cross-check everything manually before hitting submit …
```

Gold labels:

- **feedback_type** → `bug`
- **product_area** → `notifications`

### Banking intent (`banking_intent`)

A branch visit can mix a wire, a loan, and a hold. The label is the operation the core system should open.

```text
[channel] branch
[utterance] I’m here at the Riverside branch regarding the auto loan my new employer, Ferris & Riggs Group, recently approved and started funding through my Harborline business banking profile. I authorized an international wire transfer for the down payment to AutoBazaar GmbH in Germany last Friday, and I see the funds are held under an authorization but have not yet posted as a completed …
```

Gold labels:

- **intent** → `other`

### Clinic request (`clinic_request`)

Patients describe a symptom and the thing they want in the same sentence. The desk needs the request and whether it is urgent.

```text
[channel] chat
[message] When I first spoke with the clinic last month, the nurse told me that Dr. Hammond had submitted the referral for my post-op physical therapy directly to the rehab center, but I haven't received any scheduling info yet. I had a follow-up appointment planned for next week, but I missed it because of a work emergency. I know I should be doing some light exercises related to my amoxicillin …
```

Gold labels:

- **request** → `other`
- **urgent** → `yes`

### Travel request (`travel_request`)

A change, a seat, and a hotel voucher look alike in chat and trigger different inventory calls.

```text
[channel] app
[trip] QL549, terminal 2, Impora to Zavidin via Altor, departing 8 July
[message] When considering my upgraded premium seat on QL549 next week, I realized I might need assistance with accommodations due to the overnight layover in Altor — could you please confirm if a hotel voucher is included with my fare class? Also, I've already submitted my visa documents for this trip, but apologies if any details …
```

Gold labels:

- **intent** → `hotel`

### News topic (`news_topic`)

A headline has to land in a section before it is ranked. The label set is the section list.

```text
[headline] Ferry services resume as rural Eastport hosts annual "Wind Songs" festival
[lead] A service disruption left thousands stranded along the Eastport coast after mechanical issues forced ferry cancellations during last week's storm. Organizers now say the local "Wind Songs" music festival, scheduled for 5 April, will proceed as vessels return—responding to community calls to boost visitor numbers following a …
```

Gold labels:

- **topic** → `world`

### Paper field (`paper_field`)

Title and abstract in, the field the library or the reviewer queue should file.

```text
[title] A Microfluidic Reactor for Confidence-Calibrated Enzyme Cascade Synthesis
[abstract] Conventional batch synthesis of the target fluorinated ester requires prolonged stirring and gives variable conversion under laboratory conditions. We designed a segmented-flow microreactor with immobilized lipase and an inline optical detector, then compared it with the established stirred-vessel protocol using matched …
```

Gold labels:

- **field** → `chemistry`

### Sports recap (`sports_recap`)

Sport, result, and whether the result was an upset, from one recap.

```text
[recap]
At the summit of the Crystalline Peaks Circuit, the Zenith Riders and Aurora Speedsters locked horns in a championship showdown that ended in a thrilling stalemate. The high-altitude track was bone-dry, offering perfect grip, yet the relentless pace challenged tire wear across the field. Early series saw frequent lead swaps, with the Riders leveraging aggressive overtakes, while the Speedsters countered with …
```

Gold labels:

- **sport** → `motorsport`
- **result** → `draw`
- **upset** → `yes`

### Restaurant review (`restaurant_review`)

Sentiment is one label. Aspects are several: food, service, price, wait. `aspects` is multi-label.

```text
[place] Trattoria Lucca, Orvieto Crossing
[text] Compared with my last visit, our drive-through stop at Trattoria Lucca before the jazz concert was disappointing. Crisp garlic hit first, but the handmade tagliatelle was overcooked, missing that al dente bite. My serving arrived tepid, and the basil was wilted. Buon appetito felt forced this time.
```

Gold labels:

- **sentiment** → `negative`
- **aspects** → `food`

### Benefits request (`benefits_request`)

Which program the person is asking about, and whether they are asking for the status of a claim already filed.

```text
[office] Ravenwood County Benefits Department
[message] My existing unemployment claim, case number UB-55902, is pending review since I submitted all required documents last month. I provided a medical certification from Pinewood Clinic, as I had a temporary disability that delayed my job search. I recently relocated to a new apartment, so I’m uncertain if my updated address is properly recorded. I am also preparing …
```

Gold labels:

- **program** → `unemployment`
- **asking_status** → `yes`

### Screen tags (`screen_tags`)

A title needs a format and every genre that fits. `genres` is multi-label.

```text
At the bustling heart of Viridian City in 1912, Project LARK-27 bridges neighborhoods through a public library's curious new reading circles. Each episode intertwines the daily adventures of Sofina, a young nanny, and Meena, the spirited child she cares for, as they encounter a parade of inventive library contests, rooftop rain gardens, and trams packed with storytellers. Drawing from community traditions and …
```

Gold labels:

- **genres** → `kids, drama`
- **format** → `series`

## Benchmark

Exact-match accuracy on the held-out test split: 17 domains, 300 examples each, the same text and the same candidate labels for every model. [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) is the English model. [GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) is the multilingual model, scored here on English.

| Model | Avg |
|---|---:|
| **GLiNER2.5-Decide** | **60.2%** |
| GLiNER2 XL (1B) | 59.6% |
| JevK5 | 57.6% |
| GLiNER2.5-multi-Decide | 56.7% |
| SemIf (Qwen3.5-4B) | 56.4% |
| GLiFormer large-v1 | 49.0% |
| Laya Router | 46.6% |

| Split | Examples per domain | Where |
|---|---:|---|
| Development | 100 | This dataset |
| Test | 300 | Held out |

## Loading

One domain:

```python
from datasets import load_dataset

ds = load_dataset("fastino/fast-decisions", "support_intent", split="train")
```

Every domain:

```python
from datasets import load_dataset, concatenate_datasets

domains = [
    "support_intent", "support_topic", "document_type", "review_sentiment",
    "agent_handoff", "email_triage", "ticket_route", "product_feedback",
    "banking_intent", "clinic_request", "travel_request", "news_topic",
    "paper_field", "sports_recap", "restaurant_review", "benefits_request",
    "screen_tags",
]
all_ds = concatenate_datasets([
    load_dataset("fastino/fast-decisions", name, split="train") for name in domains
])
```

Exact match against [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide). Pass each head's `multi_label` flag through, otherwise `classify_text` defaults to single-label and every multi-label head with more than one gold label is scored wrong. A single-label head returns a string. A multi-label head returns a list. Both are compared as sets.

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")

hits = total = 0
for row in ds:
    for head in row["output"]["classifications"]:
        task = {
            "labels": head["labels"],
            "multi_label": head.get("multi_label", False),
        }
        pred = model.classify_text(row["input"], {head["task"]: task})
        got = pred[head["task"]]
        if isinstance(got, str):
            got = [got]
        elif isinstance(got, dict):
            got = [got["label"]]
        hits += sorted(got) == sorted(head["true_label"])
        total += 1
```

For text that is not English, load [GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) the same way.

## Details

- **What it is:** seventeen operational classification domains for [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
- **What it is not:** a reasoning exam or a copy of a public benchmark
- **Published split:** development, 100 rows per domain
- **Scored split:** test, 300 rows per domain, held out
- **License:** Apache 2.0

Names, account numbers, and amounts in the text are fictional.

## Citation

```bibtex
@misc{zaratiana2025gliner2efficientmultitaskinformation,
      title={GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface},
      author={Urchade Zaratiana and Gil Pasternak and Oliver Boyd and George Hurn-Maloney and Ash Lewis},
      year={2025},
      eprint={2507.18546},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.18546},
}
```


# Circuit Breaker Parameter Calibration Guide

**Aegis MetroHealth — Department of Dermatology & Cutaneous Oncology**  
*Metamorphic Circuit Breaker — Phase 2 Hierarchical Safety Engine*

---

## Overview

The **Metamorphic Circuit Breaker** works by running your skin lesion image through 22 different "stress tests" (metamorphic transformations), then measuring how much the AI model's diagnosis changes across those tests. If the diagnosis stays stable — you're good. If it keeps flipping — the system raises a safety flag.

The parameters below control **how sensitive** this safety system is. Getting them right means fewer false alarms for stable images and reliable catches of genuinely fragile predictions.

---

## The 6 Parameters at a Glance

| Parameter | Default | Role | Analogy |
|-----------|---------|------|---------|
| `alpha` | 0.50 | Weight: worst-performing test family | "How much does the worst subject's failure matter?" |
| `beta` | 0.35 | Weight: spread across multiple families | "Does the failure appear everywhere, or just one area?" |
| `gamma` | 0.15 | Weight: model's own confidence | "Is the model itself unsure?" |
| `tau_fam` | 0.35 | Family compromise threshold | "How bad does one family need to be to count as broken?" |
| `theta_warn` | 0.25 | Warning boundary → HALF-OPEN | "When to raise a yellow flag?" |
| `theta_trip` | 0.55 | Trip boundary → OPEN (block AI) | "When to pull the emergency brake?" |

> **Rule:** `alpha + beta + gamma` must always equal exactly **1.0**. These three weights share 100% of the formula's attention.

---

## Part 1 — The Three CBI Formula Weights

The system computes a single number called the **Composite Brittleness Index (CBI)**:

```
CBI = alpha × max_family_score  +  beta × cross_family_spread  +  gamma × (1 - confidence)
```

Think of CBI as a **score from 0.0 to 1.0** where:
- `0.0` = perfectly rock-solid prediction
- `1.0` = completely unstable, untrustworthy model output

---

### `alpha` — Peak Family Weight (default: 0.50)

**What it measures:** The worst-performing transformation family (Geometric, Photometric, or Sensor/Noise).

**Plain English:** If the model's diagnosis flips a lot when you rotate the image (Geometric family), `alpha` controls how much that single bad performance drives the overall CBI score.

**Example:**
- Geometric instability score = 0.80 (lots of flips under rotation/flip/zoom)
- With `alpha = 0.50`: contributes `0.50 × 0.80 = 0.40` to CBI
- With `alpha = 0.70`: contributes `0.70 × 0.80 = 0.56` to CBI — more sensitive to rotation issues

**When to increase `alpha`:**
- You care most about catching models that collapse under viewpoint/geometric changes (common in poorly acquired images)
- Your imaging equipment produces frequent orientation inconsistencies

**When to decrease `alpha`:**
- You trust your camera setup produces geometrically consistent images
- You're getting too many false alarms on stable images with minor geometric sensitivity

---

### `beta` — Cross-Family Spread Weight (default: 0.35)

**What it measures:** Whether instability is localized to one family or spread across multiple families.

**Plain English:** If the model fails under rotation *and* under lighting changes *and* under noise — that's much more alarming than failing under just one type. `beta` captures this "spread" of failure.

**How it's calculated:**
- Each family that scores above `tau_fam` (see below) is counted as "compromised"
- Cross-Family Spread (CFS) = (number of compromised families) / 3
- E.g., 2 families compromised → CFS = 0.667; 1 family → CFS = 0.333

**Example:**
- 2 out of 3 families are compromised → CFS = 0.667
- With `beta = 0.35`: contributes `0.35 × 0.667 = 0.23` to CBI

**When to increase `beta`:**
- Multi-family failure is a strong red flag in your clinical setting (most cases)
- You want the system to be extra cautious when failures appear in more than one family

**When to decrease `beta`:**
- You work with images where a single family routinely misbehaves (e.g., old equipment with sensor noise) and you don't want that alone to trigger alerts

---

### `gamma` — Baseline Uncertainty Weight (default: 0.15)

**What it measures:** How confident the AI model was in its *own* original prediction (before any tests).

**Plain English:** Even without doing any transformation tests, if the model says "I think it's melanoma — 54% sure", that low confidence is itself a warning sign. `gamma` adds this native uncertainty into the CBI.

**Formula component:**
```
gamma × (1 - baseline_confidence)
```
- Model 95% confident → contributes `gamma × 0.05` (small)
- Model 54% confident → contributes `gamma × 0.46` (larger)

**Example:**
- Confidence = 72% → `gamma × 0.28 = 0.15 × 0.28 = 0.042`
- Confidence = 51% → `gamma × 0.49 = 0.15 × 0.49 = 0.074`

**When to increase `gamma`:**
- You want low-confidence AI predictions to be flagged even if the metamorphic tests are stable
- Your clinical protocol requires a physician second-opinion for any borderline model output

**When to decrease `gamma`:**
- The model is trained on a specific narrow domain and consistently produces moderate (not high) confidence — you don't want routine cases flagged just for being 65% confident

> **Preset profiles:** The Calibration Studio offers pre-tuned weight profiles:
> - **Balanced (default):** `0.50 / 0.35 / 0.15` — recommended for ISIC dermoscopy images
> - **Sensitivity-First:** `0.40 / 0.40 / 0.20` — flags more cases, fewer missed detections
> - **Specificity-First:** `0.60 / 0.30 / 0.10` — fewer false alarms, strict on only severe instability

---

## Part 2 — The Family Threshold

### `tau_fam` — Family Compromise Threshold (default: 0.35)

**What it measures:** The minimum instability score that marks a transformation family as "broken" (compromised).

**Plain English:** Imagine each of the 3 test families (Geometric, Photometric, Sensor/Noise) is a student taking a test. `tau_fam` is the **pass/fail grade**. Any family scoring below this passes; any scoring above this fails.

**How family scores are computed (simplified):**
1. Each individual test (e.g., "rotate 15 degrees") is scored: did the prediction flip? By how much did the severity matter?
2. All tests within a family are averaged → family score IR_k (ranges 0.0 to 1.0)
3. If IR_k ≥ `tau_fam` → that family is "compromised"

**Example:**
- Geometric family IR_k = 0.40
- With `tau_fam = 0.35`: **Compromised** (0.40 ≥ 0.35) → counts toward cross-family spread
- With `tau_fam = 0.45`: **Not compromised** (0.40 < 0.45) → passes

**When to lower `tau_fam`** (make it easier to fail):
- You want the system to be extra sensitive — even small family instability should be flagged
- Try `tau_fam = 0.20` to make the system trip on borderline cases

**When to raise `tau_fam`** (make it harder to fail):
- You're getting too many OPEN/HALF-OPEN alerts on images your clinicians consider stable
- Try `tau_fam = 0.50` to only flag families with genuinely major instability

> **Important:** `tau_fam` affects how many families get counted as compromised, which directly feeds into `beta`'s cross-family spread calculation. Lowering `tau_fam` makes the cross-family spread (CFS) larger.

---

## Part 3 — The State Transition Thresholds

These two parameters define the **boundaries** between the three circuit breaker states.

```
CBI range           →   State               →   Action
──────────────────────────────────────────────────────────────────────────────
0.00  to  theta_warn    CLOSED             Auto-approve, no human needed
theta_warn  to  theta_trip   HALF-OPEN     Approve with amber warning to doctor
theta_trip  to  1.00    OPEN               Block AI prediction, mandatory doctor review
```

---

### `theta_warn` — Warning Threshold (default: 0.25)

**What it measures:** The CBI value at which the system switches from fully auto-approved (CLOSED) to monitoring mode (HALF-OPEN).

**Plain English:** This is the **yellow flag** line. Below it, the system trusts the AI completely. Above it, the system says "something's a little off — let a doctor know."

**Example:**
- CBI = 0.18 → CLOSED (auto-approved, no action needed)
- CBI = 0.30 → HALF-OPEN (result forwarded to Dr. Sarah Jenkins for a quick review)

**When to lower `theta_warn`** (catch more cases):
- You want even mild model sensitivity to trigger a doctor's awareness
- High-stakes patient population (elderly patients, atypical lesions, history of melanoma)
- Try `theta_warn = 0.15` for maximum caution

**When to raise `theta_warn`** (fewer interruptions):
- Your doctors are overwhelmed with HALF-OPEN queue cases that turn out to be clearly benign
- Stable, well-controlled imaging environment
- Try `theta_warn = 0.35` to reduce physician workload

---

### `theta_trip` — Trip Threshold (default: 0.55)

**What it measures:** The CBI value at which the circuit breaker **fully trips** — blocking the automated AI output entirely and demanding a physician override.

**Plain English:** This is the **emergency brake**. When CBI crosses this line, the system refuses to release the AI's diagnosis to the patient. A dermatologist must review the image directly and record their own verdict.

**Example:**
- CBI = 0.48 → HALF-OPEN (between 0.25 and 0.55) — amber warning, doctor is notified
- CBI = 0.62 → OPEN (above 0.55) — AI diagnosis blocked, mandatory physician intervention

**When to lower `theta_trip`** (trip more aggressively):
- You want stronger protection — even moderately unstable models should be blocked
- Regulatory environment requiring conservative AI use
- Try `theta_trip = 0.40` for a very sensitive breaker

**When to raise `theta_trip`** (allow more through automatically):
- The OPEN queue is too full and most cases turn out to be false alarms
- Your model is particularly robust and rarely exhibits true multi-family collapse
- Try `theta_trip = 0.70` for a more permissive system

> **Key rule:** `theta_warn` must always be **less than** `theta_trip`. If they are equal or reversed, the HALF-OPEN state disappears.

---

## Summary: How the Parameters Interact

```
                     ┌──────────────────────────────────────────┐
      Image Input    │    22 Metamorphic Tests run              │
         │           │    (Geometric / Photometric / Sensor)    │
         ▼           └──────────────────────────────────────────┘
  ┌─────────────┐                      │
  │  Baseline   │              Level 1 Normalization
  │  Inference  │             Score_m per transform type
  └─────────────┘                      │
         │                    Level 2 Normalization
   c0 = confidence           IR_k per family (tau_fam check)
   used by gamma                        │
         │                    ┌─────────┴──────────┐
         │                 peak IR_k            CFS score
         │               (alpha weight)       (beta weight)
         │                    └─────────┬──────────┘
         └──────────────────────────────┘
                              │
                    CBI = alpha × max_IR_k
                          + beta × CFS
                          + gamma × (1 - c0)
                              │
                   ┌──────────┼──────────┐
                CBI < 0.25   0.25-0.55  CBI ≥ 0.55
                   │             │           │
                CLOSED      HALF-OPEN      OPEN
              Auto-approve   Doctor alert  AI blocked
```

---

## Practical Calibration Scenarios

### Scenario A — Reducing false alarms on stable ISIC images
The model was trained on ISIC 2019 and produces many HALF-OPEN results even on clear nevus images.

**Try:**
```
theta_warn  = 0.35   (raise the yellow flag line)
tau_fam     = 0.45   (require higher instability to count a family as broken)
alpha       = 0.60   (focus on peak family only, reduce spread sensitivity)
beta        = 0.25
gamma       = 0.15
```

---

### Scenario B — Maximum sensitivity for high-risk melanoma screening
You're screening a high-risk population and want the system to catch every possible instability.

**Try:**
```
theta_warn  = 0.15   (flag even mildly unstable predictions)
theta_trip  = 0.35   (trip aggressively)
tau_fam     = 0.20   (even low family instability counts)
alpha       = 0.45
beta        = 0.40   (weight cross-family spread heavily)
gamma       = 0.15
```

---

### Scenario C — Testing with non-dermoscopy smartphone images
Smartphone images routinely trigger Geometric instability (rotation, scale) due to inconsistent capture angles. You don't want geometric noise to dominate.

**Try:**
```
alpha       = 0.35   (reduce peak family weight — geometric family will dominate)
beta        = 0.45   (weight spread more — if photometric and noise are stable, don't trip)
gamma       = 0.20   (give model confidence slightly more say)
tau_fam     = 0.50   (geometric family needs to be very bad to be counted as compromised)
theta_warn  = 0.30
theta_trip  = 0.60
```

---

## Notes on the Severity Weight System

Within each family, individual tests carry different **severity weights**. This means not all flips are treated equally:

| Severity | Weight | What it means |
|----------|--------|---------------|
| Mild | **1.0** | Easy test — if the model flips here, it's a serious problem |
| Moderate | **0.6** | Medium difficulty test — a flip is concerning |
| Severe | **0.3** | Hard test — a flip under extreme transformation is expected, less alarming |

**Why?** If a model changes its diagnosis when you rotate an image by 15 degrees (mild), that's much more alarming than it changing its mind under an extreme 135-degree rotation with added heavy noise (severe). The weights encode this clinical intuition.

These severity weights are **fixed by the framework design** and cannot be adjusted in the Calibration Studio — they are part of the Phase 2 hierarchical normalization architecture.

---

## Quick Reference Card

```
alpha + beta + gamma = 1.0  (always)
tau_fam controls who "fails" per family
theta_warn < theta_trip  (always)

LOWER theta values  →  MORE cases forwarded to doctor
HIGHER tau_fam      →  FEWER families counted as compromised
HIGHER alpha        →  PEAK family drives more of the CBI score
HIGHER beta         →  SPREAD across families drives more of the CBI score
```

---

*This guide is intended for clinical engineering staff and consultant dermatologists configuring the AI safety system. Parameter changes take immediate effect on the next screening run.*

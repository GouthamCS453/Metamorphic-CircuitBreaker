# Metamorphic Circuit Breaker for Vision Models

## Problem Statement and Proposed Idea

### 1. Problem

Vision-based deep learning models can produce confident but incorrect predictions when an input undergoes small changes. A metamorphic testing approach can expose this instability by applying transformations that are expected to preserve the semantic meaning of the image and checking whether the model's behavior remains consistent.

The difficulty in a practical circuit-breaker system is that a changed prediction does not automatically mean that the model is faulty. The transformation itself may have changed the semantic content of the image, or the transformation may be inappropriate for the particular dataset or application.

**Example:** If an original image is classified as **Cat** and a transformed image is classified as **Dog**, there are two possible explanations:

1. The transformation was inappropriate or too strong.
2. The transformation preserved the meaning and the model failed to remain robust.

### 2. Core Problem to Solve

**How can a circuit-breaker mechanism distinguish between a transformation-specific failure and a genuine model-level robustness failure?**

Absolute causal determination is generally not possible from a single transformed input without an external oracle. Therefore, the proposed idea is to infer the most likely cause from the pattern of failures across multiple semantically validated transformations.

### 3. Proposed Idea

Instead of treating every metamorphic transformation as an independent binary test, classify transformations into meaningful families and analyze the model's behavior across multiple transformations within each family.

| Transformation Family | Examples |
|---|---|
| Geometric | Rotation, horizontal flip, translation, scaling, perspective |
| Photometric | Brightness, contrast, saturation, exposure, color changes |
| Sensor / Noise | Gaussian noise, blur, compression, sensor noise |

### 4. Transformation-Level Analysis

First, examine whether the instability is isolated to a particular transformation.

**Example:**

- Rotation 5° → Stable
- Rotation 10° → Failure
- Horizontal flip → Stable
- Translation → Stable

This suggests transformation-specific instability. The system should not immediately conclude that the model is generally faulty. The particular transformation, its severity, or its suitability for the application should first be questioned.

### 5. Transformation-Family Analysis

Next, examine whether multiple transformations belonging to the same family produce failures.

**Example:**

- Rotation → Failure
- Flip → Failure
- Translation → Failure
- Scaling → Failure
- Brightness → Stable
- Contrast → Stable
- Saturation → Stable

Repeated failures across several geometric transformations provide much stronger evidence of a geometric robustness weakness in the model than a failure from one transformation alone.

### 6. Cross-Category Analysis

If failures occur across several independent transformation families, the evidence becomes stronger that the model has a broader robustness problem rather than a problem associated with one transformation.

**For example:**

- Geometric → Multiple failures
- Photometric → Multiple failures
- Sensor/Noise → Multiple failures

This would indicate broad model instability and could justify activating the circuit breaker and redirecting the case to human input.

### 7. Important Addition: Transformation Severity

Each transformation should ideally be tested at several severity levels. For example, instead of testing only rotation, test **1°, 3°, 5°, 10°, 15°, and 20°**.

If the model is stable up to 10° and fails only at 20°, the transformation may simply have crossed a meaningful robustness boundary. If the model fails at 1°–3°, that is much stronger evidence of model fragility.

### 8. Transformation Qualification

The transformation families cannot simply be assumed to be valid. Before runtime deployment, candidate transformations should be qualified for the specific dataset and application.

For example, a rotation that is semantically harmless for one dataset may not be harmless for another. Similarly, a large brightness, saturation, crop, or rotation change may alter important information.

Therefore, the system should maintain an approved transformation set with suitable parameter ranges. Runtime diagnosis should primarily use these validated transformations.

### 9. Proposed Decision Logic

| Observed Behavior | Interpretation |
|---|---|
| One transformation fails | Transformation-specific anomaly; model fault not yet established. |
| Several transformations in one family fail | Evidence of a family-level model robustness weakness. |
| Several independent families fail | Evidence of broad model instability. |
| Failure occurs only at high severity | Possible transformation/severity boundary; investigate before blocking. |
| Semantic validity is uncertain | Cause is ambiguous; prefer human verification rather than automatic attribution. |

### 10. Circuit-Breaker Concept

The circuit breaker should therefore not simply trigger whenever one metamorphic test fails. It should use the aggregated evidence to decide how strongly the model should be trusted.

**Conceptual flow:**

> Input → Original Prediction → Generate Approved Transformations → Run Model on Transformations → Compare Predictions → Analyze Failures by Transformation → Aggregate by Family → Evaluate Severity → Estimate Fault Attribution → Decide Action

**Possible actions are:**

- **Normal:** Model behavior remains stable.
- **Monitor:** Isolated or weak instability is detected.
- **Circuit break:** Repeated family-level or cross-family instability provides strong evidence that the model should not be trusted.
- **Human input:** The evidence is strong but the cause remains ambiguous, or the application is sufficiently safety-critical that automated attribution is not adequate.

### 11. Central Research Idea

The key idea is to use the **pattern of metamorphic failures—not a single failed transformation—to attribute instability**.

An isolated failure suggests a transformation-specific problem. Repeated failures across multiple independently validated transformations in the same category suggest a model-level weakness in that type of robustness. Failures across multiple categories suggest broader model instability.

This makes the proposed mechanism more conservative and useful than a simple rule of:

> **Prediction changed → Circuit breaker**

It also acknowledges that absolute causal attribution is not possible without an external oracle.

### 12. Proposed Research Direction

The concept can therefore be framed as:

# Metamorphic Fault Attribution for Vision Model Circuit Breaking

The research objective would be to investigate whether hierarchical analysis of semantically validated transformation families can improve the ability of a metamorphic circuit breaker to distinguish transformation-specific anomalies from genuine model robustness failures and make safer decisions about when to redirect a prediction to human verification.

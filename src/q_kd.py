from qdsl import tf, single, multi, num, T

import math

K = "kd"


def softmax(z, temp=1.0):
    e = [math.exp(x / temp) for x in z]
    s = sum(e)
    return [x / s for x in e]


# ---------------------------------------------------------------- calc data (computed, never typed by hand)
LOGITS = (2, 1, 0)
TEMP = 2
q_T2 = softmax(LOGITS, TEMP)
q_T1 = softmax(LOGITS, 1)
q_mul = softmax([x * TEMP for x in LOGITS])          # mistake: multiplying by T instead of dividing
exps = [math.exp(x / TEMP) for x in LOGITS]
assert abs(q_T2[0] - 0.5065) < 5e-5 and abs(sum(q_T2) - 1) < 1e-12

CONF = (9, 2, 1)                                     # a very confident teacher
conf_1 = softmax(CONF, 1)
conf_3 = softmax(CONF, 3)
assert conf_1[0] > 0.998 and conf_3[0] < 0.9

# ------------------------------------------------------------------ true / false
tf(K, "A teacher's softened output distribution tells the student which wrong classes resemble the correct one, which a one-hot label cannot express.", True,
   """A one-hot label puts probability 0 on every wrong class and so treats them all as equally wrong. The teacher's soft targets keep the relative probabilities of the wrong classes: a picture of a car is far more likely to be confused with a truck than with a carrot.

Hinton popularised the name “dark knowledge” for this hidden similarity structure (the paper itself speaks of the relative probabilities of the incorrect answers); it makes each training example carry much more information than its hard label.""")

tf(K, "If the student is trained only on the teacher's soft targets, with no hard-label term in the loss, the examples used for distillation need no ground-truth labels.", True,
   """The targets come from the teacher's outputs, so any inputs can be used, including large pools of unlabeled data (Hinton et al. call the set of such inputs the transfer set).

Ground-truth labels are needed only if the loss also contains the hard-label cross-entropy term (weight 1 − α).""")

tf(K, "A student distilled with temperature T = 4 should also be run at T = 4 when it makes predictions on new inputs, so that its outputs are as soft as the targets it was trained to match.", False,
   """The raised temperature is a training device. After distillation the student is used with the ordinary softmax, T = 1.

Dividing the logits by any T > 0 never changes the arg max, so the predicted class would be the same, but T = 1 is the setting in which the student was also trained on the hard labels and in which its probabilities are meant to be read.""")

# ------------------------------------------------------------------ single answer
single(K, "In the standard distillation objective, the soft-target term is multiplied by T², where T is the softmax temperature. What is the purpose of this factor?",
       ("Soft-target gradients shrink roughly like 1/T², so the factor keeps them comparable to the hard-label gradients as T changes",
        "For a student logit z_{i} the soft-loss gradient is (p_{i} − q_{i})/T, where p and q are the student's and the teacher's softened probabilities. At large T the difference p_{i} − q_{i} itself shrinks like 1/T, so the gradient falls like 1/T². Multiplying by T² keeps the soft and hard terms in proportion, so the weight α need not be re-tuned when T changes."),
       [("It renormalises the temperature-scaled probabilities, which would otherwise no longer sum to one over the classes",
         "The temperature softmax is normalised for every T, because its denominator sums exp(z_{j}/T) over all classes. T² multiplies the loss, not the probabilities."),
        ("It scales the update by the size gap between the two networks, so that a much smaller student takes proportionally larger steps",
         "Model sizes appear nowhere in the objective. The factor depends only on the temperature, and it compensates for what the temperature does to the gradient magnitude."),
        ("It makes the KL divergence symmetric, so that swapping the roles of teacher and student gives the same loss",
         "A constant factor cannot remove the asymmetry of KL(q ‖ p); the loss still treats the teacher's distribution as the reference that the student must match.")],
       """Gradient of the soft term with respect to a student logit z_{i}: (1/T)(p_{i} − q_{i}) ≈ (z_{i} − v_{i}) / (N T²) for large T and zero-mean logits. Here p and q are the student's and the teacher's probabilities at temperature T, v are the teacher's logits and N is the number of classes.

Because it falls like 1/T², multiplying the soft term by T² keeps its contribution roughly constant. T can then be varied during experiments without re-balancing the soft term against the hard-label term.""")

single(K, T("A pre-trained teacher produces the logits z = (${z1}, ${z2}, ${z3}) for a training image. At T = 1 this gives the output (${a:.4f}, ${b:.4f}, ${c:.4f}), which is almost one-hot. Why does distillation raise the temperature before the teacher's logits are turned into soft targets?",
            z1=CONF[0], z2=CONF[1], z3=CONF[2], a=conf_1[0], b=conf_1[1], c=conf_1[2]),
       ("At T = 1 the tiny wrong-class probabilities barely affect the loss; a larger T lifts them so the student also learns their relative sizes",
        T("Dividing the logits by T = 3 gives (${a:.3f}, ${b:.3f}, ${c:.3f}): the wrong classes now receive probabilities large enough to influence the loss (${b:.3f} and ${c:.3f} instead of ${b1:.4f} and ${c1:.4f}), so the student is rewarded for matching their relative sizes, which is the similarity information that a nearly one-hot target hides.", a=conf_3[0], b=conf_3[1], c=conf_3[2], b1=conf_1[1], c1=conf_1[2])),
       [("A larger T multiplies the logits, which makes the teacher's top class even more dominant for the student to copy",
         "The temperature divides the logits. A larger T therefore flattens the distribution; multiplying the logits would produce the sharper, nearly one-hot targets that distillation tries to avoid."),
        ("A larger T makes the teacher itself more accurate, because scaling the logits removes noise from its predictions",
         "Dividing by T > 0 is a monotone rescaling: the ranking of the classes and the arg max stay the same, so the teacher's accuracy is unchanged. Only the softness of the distribution changes."),
        ("A small student cannot output probabilities as extreme as 0.9988, so the targets must first be flattened until it can reach them",
         "Capacity is not the issue: a small network can produce arbitrarily large logits. The reason is information content: extreme targets hide the relative probabilities of the wrong classes.")],
       """Soft targets: q_{i} = exp(z_{i}/T) / ∑_{j} exp(z_{j}/T). T = 1 is the ordinary softmax; larger T gives a flatter distribution (higher entropy).

A flatter distribution makes the small probabilities of the wrong classes large enough to matter in the loss, and these relative probabilities are the “dark knowledge” of the teacher.""")

single(K, "Which objective is used to train the student in standard Hinton-style distillation of a pre-trained teacher?",
       ("A weighted sum of T²·KL between teacher and student softmaxes at temperature T, and cross-entropy with the true label at T = 1",
        "The soft term transfers the teacher's full output distribution; the hard term keeps the student anchored to the ground truth. The weighting α balances the two."),
       [("The squared distance between the teacher's weight matrices and the student's, summed over all layers",
         "The two networks generally have different sizes, so their weights do not even line up. Distillation matches output behaviour (the output distributions), not parameters."),
        ("Cross-entropy against the teacher's arg max class, used as a hard pseudo-label in place of the true label",
         "Keeping only the arg max throws away the soft distribution, which is the whole point of distillation. It would reduce to ordinary training on (possibly wrong) hard labels."),
        ("The sum of the teacher's and the student's losses, back-propagated through both networks so that they improve together",
         "In the standard recipe the teacher is already trained and its weights stay fixed, so there is no teacher loss to optimise. Only the student's parameters are updated.")],
       """L = α · T² · KL( softmax(v/T) ‖ softmax(z/T) ) + (1 − α) · CE( y, softmax(z) )

with v the teacher logits (fixed) and z the student logits. At test time the student is used with T = 1.""")

single(K, "In sequence-level knowledge distillation for neural machine translation (Kim & Rush, 2016), what is the student trained on?",
       ("Source sentences paired with the teacher's own beam-search translations, used as if they were the references",
        "The teacher decodes the training sources, and its best hypothesis becomes the new target for ordinary maximum-likelihood training of the student."),
       [("Source sentences paired with the teacher's complete next-word distribution at every target position",
         "That describes word-level distillation, which matches the teacher's per-token distributions. The sequence-level variant uses only the decoded sentence."),
        ("The original human references only; the teacher is used afterwards to re-rank the student's n-best list at test time",
         "Re-ranking is a decoding technique and does not change the student's training data. Sequence-level distillation changes the targets the student is trained on."),
        ("The teacher's hidden states, which the student has to reproduce layer by layer while reading the same source sentence",
         "Matching intermediate representations is hint- or feature-based distillation, which is a different method. Sequence-level KD needs only the teacher's output sentences.")],
       """Word-level KD: cross-entropy between the teacher's and the student's next-token distributions at each position.

Sequence-level KD: approximate the teacher's distribution over whole translations by its most likely output. Decode the training sources with beam search, then train the student on (source, teacher output) pairs as usual. The student is typically smaller and faster, at a small cost in quality.""")

# ------------------------------------------------------------------ multiple answer
multi(K, "Distillation softens probabilities with the temperature softmax q_{i} = exp(z_{i}/T) / ∑_{j} exp(z_{j}/T). Which of the following statements about T are correct? Select all that apply.",
      [("With T = 1 this is the ordinary softmax",
        "Dividing the logits by 1 changes nothing."),
       ("As T grows very large, the distribution approaches the uniform distribution over the classes",
        "All ratios z_{i}/T tend to 0, so every exp(·) tends to 1 and every probability tends to 1/K, where K is the number of classes."),
       ("As T tends to 0, the distribution approaches a one-hot vector on the class with the largest logit (if that logit is unique)",
        "Small T magnifies the logit gaps, so the largest logit takes all of the probability mass."),
       ("For any T > 0 the ordering of the class probabilities, and hence the arg max, is the same as at T = 1",
        "Dividing all logits by the same positive number is monotone. Temperature changes how sharp the distribution is, not which class comes first.")],
      [("Choosing T < 1 gives softer targets than the ordinary softmax",
        "The opposite: T < 1 amplifies the differences between logits and gives sharper (more peaked) targets. Soft targets need T > 1."),
       ("In standard distillation T is a trainable parameter that gradient descent adjusts together with the student's weights",
        "T is a hyperparameter chosen by the practitioner (for example by validation); it is not learned in the usual formulation.")],
      """Temperature softmax: T → 0 gives a one-hot, T = 1 the usual softmax, T → ∞ the uniform distribution. The arg max never changes.

Distillation uses T > 1 on the teacher's and the student's logits, so that the relative probabilities of wrong classes become visible.""")

multi(K, "Which of the following hold for the teacher and the student in Hinton-style distillation? Select all that apply.",
      [("The teacher can be an ensemble of networks whose averaged predictions are distilled into one student",
        "Compressing an ensemble into a single model was a central use case: the student inherits much of the ensemble's accuracy at the cost of one network."),
       ("The teacher needs no special training: an ordinarily trained network is used, and the temperature is raised only when its logits are turned into soft targets",
        "The teacher's softmax is trained at T = 1 as usual; T > 1 is applied afterwards to its logits (and to the student's) for the distillation loss."),
       ("While the student is being trained, the teacher's weights stay fixed and only the student is updated",
        "In the standard offline recipe the teacher is a frozen source of targets, which can be computed once or on the fly.")],
      [("The student has to use the same architecture as the teacher, because otherwise its outputs cannot be matched to the soft targets",
        "Only the output distributions must be compatible (same label set). The student is typically a smaller or different architecture, which is the whole point of compression."),
       ("A distilled student is guaranteed to reach the teacher's accuracy, because it is trained to reproduce the teacher's outputs",
        "Nothing guarantees this. A smaller student usually ends up slightly below its teacher; the trade-off is a good part of the accuracy for much less compute and memory.")],
      """Teacher: trained first (single model or ensemble), then frozen. Student: smaller or different architecture, trained on the teacher's soft targets (and optionally hard labels).

Goal: compression and speed with only a small loss of accuracy; at inference only the student is run.""")

multi(K, "Select all correct statements about distillation methods for NLP models.",
      [("Word-level KD compares the student's and the teacher's next-token distributions at every target position, so it needs the teacher's output probabilities",
        "It is the token-level analogue of Hinton's soft-target loss, applied at each decoding step."),
       ("Sequence-level KD needs only sentences generated by the teacher, so the teacher can even be a black box that just returns translations",
        "The student is trained with ordinary cross-entropy on the teacher's outputs, so no probabilities or internal states of the teacher are required."),
       ("DistilBERT halves the number of Transformer layers of BERT (6 instead of 12) and is trained with, among other terms, a soft-target loss from BERT",
        "Its training objective combines the distillation loss on the teacher's soft predictions, the usual masked-language-modelling loss and a cosine loss on hidden states; it is much smaller and faster while keeping most of BERT's accuracy.")],
      [("Sequence-level KD hands the student the teacher's probabilities for the alternative words at each position, just as word-level KD does",
        "The student sees only the single decoded sentence, i.e. hard targets produced by the teacher. The probabilities of alternatives are not part of its training data."),
       ("DistilBERT was obtained by pruning the smallest weights of a trained BERT and fine-tuning, so no distillation loss is involved",
        "DistilBERT is a smaller network trained by distillation (initialised from every other BERT layer). Pruning removes weights from an existing model and is a different compression technique.")],
      """Word-level KD: match the teacher's distribution at each position (needs probabilities).

Sequence-level KD: train on the teacher's decoded outputs (needs only sentences).

DistilBERT: 6-layer student of 12-layer BERT, about 40% smaller and 60% faster while retaining about 97% of BERT's language-understanding performance.""")

multi(K, "Which of the following claims connect distillation correctly to other concepts? Select all that apply.",
      [("Training on the soft targets with cross-entropy gives the same student gradients as minimising KL(teacher ‖ student), because the teacher's entropy does not depend on the student",
        "CE(q, p) = H(q) + KL(q ‖ p). The teacher is fixed, so H(q) is a constant with respect to the student's parameters."),
       ("At a high temperature, with zero-mean logits, the soft-target gradient reduces to a scaled difference of logits, so distillation approaches matching logits as in Ba & Caruana",
        "For large T the gradient with respect to a student logit z_{i} is approximately (z_{i} − v_{i}) / (N T²), where v is the teacher's logit and N the number of classes: the gradient of a squared logit difference.")],
      [("Teacher forcing is just distillation with T = 1: the ground-truth previous token plays the role of the teacher's soft target",
        "Teacher forcing feeds the reference token as the decoder's next input within a single model. Distillation uses a second model's output distribution as the training target. The two meanings of “teacher” are unrelated."),
       ("Distillation compresses a model by pruning the smallest-magnitude weights of the teacher, which yields the student",
        "That describes pruning. In distillation the student is a separately parameterised (usually smaller) network trained to imitate the teacher's outputs."),
       ("Soft targets add nothing over hard labels whenever the teacher classifies every transfer example correctly",
        "Even for correctly classified examples the soft targets carry the teacher's relative probabilities of the wrong classes, which hard labels discard.")],
      """CE with soft targets = teacher entropy (constant) + KL(teacher ‖ student).

High-temperature limit: distillation ≈ regression of the student's logits onto the teacher's logits (zero-meaned), which was the earlier logit-matching approach.

Not to be confused with teacher forcing (a decoder-input trick) or pruning (removing weights).""")

# ------------------------------------------------------------------ computation
num(K, T("A teacher network outputs the logits z = (${z1}, ${z2}, ${z3}) for three classes. Soft targets are computed with the temperature softmax q_{i} = exp(z_{i}/T) / ∑_{j} exp(z_{j}/T). What probability does the first class receive at T = ${t}?",
         z1=LOGITS[0], z2=LOGITS[1], z3=LOGITS[2], t=TEMP),
    q_T2[0],
    T("""1. Divide the logits by T = ${t}: z/T = (${a}, ${b}, ${c}).
2. Exponentiate: (${e1:.4f}, ${e2:.4f}, ${e3:.4f}); the sum is ${tot:.4f}.
3. q_{1} = ${e1:.4f} / ${tot:.4f} ≈ **${q1:.4f}**, and the full vector is (${q1:.4f}, ${q2:.4f}, ${q3:.4f}).

For comparison, T = 1 gives (${r1:.4f}, ${r2:.4f}, ${r3:.4f}): the higher temperature flattens the distribution. Multiplying by T instead of dividing would give ${m:.4f}, a sharper distribution.""",
      t=TEMP, a=LOGITS[0] / TEMP, b=LOGITS[1] / TEMP, c=LOGITS[2] / TEMP,
      e1=exps[0], e2=exps[1], e3=exps[2], tot=sum(exps),
      q1=q_T2[0], q2=q_T2[1], q3=q_T2[2],
      r1=q_T1[0], r2=q_T1[1], r3=q_T1[2], m=q_mul[0]),
    tol=0.005, hint="a decimal; 2–4 places is fine", disp=T("${q1:.4f}", q1=q_T2[0]))

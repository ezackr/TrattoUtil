# TrattoUtil

This repository acts as a separate utilities package for the Tratto project. 

# Data

This repository re-formats the original Oracles Dataset and Tokens Dataset from the Tratto Project into a more palatable format for the CodeLLaMa model. Each dataset is pre-processed separately into similar (but slightly different) formats, including a "prompt" and "label" field.

## Oracles Dataset

The Oracles Dataset from the Tratto project is processed into the new format:

```
[methodJavadoc]
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
assertTrue([oracle]);
```

For example:

```
/**
 * Returns the current count.
 *
 * @param checkpoint must not be null
 * @return the current count
 */
public int getCount(MyObject checkpoint) {
}

// "@param checkpoint must not be null" assertion
assertTrue(checkpoint != null);
```

At training time, the model is given the prompt:

```
[methodJavadoc]
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion

```

and attempts to decode the label (output):

```
assertTrue([oracle]);
```

### Empty Oracle

If an oracle is "empty" (that is, it is not possible to generate a corresponding assertion), then the corresponding label is:

```
// No assertion 
```

## Tokens Dataset

The Tokens Dataset from the Tratto Project is processed into a similar format as the Oracles Dataset, with a few augmentations, including a retrieval-based component. 

Firstly, each oracle is split into multiple datapoints, corresponding to each token. For example, in the Oracles Dataset, the oracle `checkpoint != null` corresponds to a single datapoint. However, in the Tokens Dataset, this oracle is split into 3 datapoints, corresponding to: `checkpoints`, `checkpoints !=`, and `checkpoints != null`. An additional comment is added before the assertion statement, listing all possible next tokens. Specifically, the input now has the format:

```
/**
 * [methodJavadoc]
 *
 * [javadocTags]
 */
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
// Next possible tokens: [nextPossibleTokens]
assertTrue([oracleSoFar]
```

where the list `[nextPossibleTokens]` is shuffled to avoid learning incorrect behaviors. 

At training time, the model decodes the next token, without any hard-coded restrictions on the possible predicted tokens. This means that even if `+` is not in the list of allowed tokens, it may still be predicted. However, at inference time, the model returns the "most-likely" next token (as determined by the output probabilities) from the list of possible tokens. 

### Retrieval

If the current `[oracleSoFar]` is `this.`, then there may be several possible methods in the list `[nextPossibleTokens]`. However, the prompt does not include any information about these methods, which makes it difficult for the model to determine their utility. To compensate, we prepend these methods and their Javadoc to the prompt, in the format,

```
/**
 * [otherMethodJavadoc]
 *
 * [otherMethodJavadocTags]
 */
[otherMethodModifiers] [otherMethodMethodSignature] {
}

/**
 * [methodJavadoc]
 *
 * [javadocTags]
 */
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
// Next possible tokens: [nextPossibleTokens]
assertTrue([oracleSoFar]
```

# TrattoUtil

This repository acts as a separate utilities package for the Tratto project. 

# Data

This repository contains scripts in `preprocess.py` to re-format the original Oracles Dataset from the Tratto project to use a format that is more conducive to the CodeLLaMa model.

The original `oracles-dataset` from the Tratto project processed to have the new format:

```
/**
 * [methodJavadoc]
 * [javadocTags]
 */
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
assertTrue([oracle]);
```

For example:

```
/**
 * Returns the current count
 * @param checkpoint must not be null
 * @return the current count
 */
public int getCount(MyObject checkpoint) {
}

// "@param checkpoint must not be null" assertion
assertTrue(checkpoint != null);
```

## Prompt

For prompting, the input sequence should be trimmed to the form:

```
/**
 * [methodJavadoc]
 * [javadocTags]
 */
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
assertTrue(
```

where the model then decodes the corresponding assertion. 

### Retrieval

The Tratto symbolic module reduces the number of possible predicted tokens by providing the set of compilable tokens that may follow the current sequence. For example, if the current decoded oracle is `this.`, then the `+` operator cannot be the next token, as this would create an uncompilable oracle. 

Furthermore, if the current decoded oracle is `this.`, then there may be several possible methods that may be the next token (e.g. `this.getModifiers()`). However, the current prompt does not include any information about these methods. To compensate, we prepend the possible methods to the prompt,

```
/**
 * [otherMethodJavadoc]
 * [otherMethodJavadocTags]
 */
[otherMethodModifiers] [otherMethodMethodSignature] {
}

/**
 * [methodJavadoc]
 * [javadocTags]
 */
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
assertTrue([oracleSoFar]
```

where `[oracleSoFar]` denotes the current decoded oracle (e.g. `this.`).

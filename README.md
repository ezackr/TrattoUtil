# TrattoUtil

This repository acts as a separate utilities package for the Tratto project. 

## Data

The `oracles-dataset` from the Tratto project is pre-processed to have the form:

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

### Prompt

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

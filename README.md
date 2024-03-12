# TrattoUtil

This repository acts as a separate utilities package for the Tratto project. 

# Setup

After cloning the TrattoUtil repository, create a `path/to/TrattoUtil/dataset` directory to store the Oracles and Tokens dataset. Paste both the oracles-dataset and tokens-dataset directories into the dataset directory. The layout should resemble:

```markdown
- TrattoUtil
    - dataset
        - oracles-dataset
          - train
          - validation
        - tokens-dataset
          - train
          - validation
```

Then, run the commands:
```bash
python reformat.py oracles train
python reformat.py oracles validation
python reformat.py tokens train
python reformat.py tokens validation
```
to generate each re-formatted dataset JSON file.

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

At inference time, the model is given the prompt:

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
[methodJavadoc]
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
// Next possible tokens: [nextPossibleTokens]
assertTrue([oracleSoFar][nextToken]
```

where the list `[nextPossibleTokens]` is shuffled to avoid learning incorrect behaviors, and `[nextToken]` is the next token in the oracle (and an element of the list `[nextPossibleTokens]`).

At inference time, the model is given the prompt:
```
[methodJavadoc]
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
// Next possible tokens: [nextPossibleTokens]
assertTrue([oracleSoFar]
```

and attempts to decode the label (output):
```
[nextToken]
```

The model decodes the token without any symbolic guidance or hard-coded restrictions on the possible predicted tokens. This means that even if `+` is not in the list of allowed tokens, it may still be predicted. However, at inference time, the model returns the "most-likely" next token (as determined by the output probabilities) from the list of possible tokens. 

### Retrieval

There are possible instances where external information may help inform the pre-trained model regarding the next possible oracle. For example, if the current `[oracleSoFar]` is `this.`, then there may be several possible methods in the list `[nextPossibleTokens]`. However, the prompt does not include any information about these methods, which makes it difficult for the model to determine their utility. To compensate, we prepend these methods and their Javadoc to the prompt, in the format,

```
[otherMethodJavadoc]
[otherMethodModifiers] [otherMethodMethodSignature] {
}

[methodJavadoc]
[modifiers] [methodSignature] {
}

// "[targetTag]" assertion
// Next possible tokens: [nextPossibleTokens]
assertTrue([oracleSoFar]
```

# Training with LLaMa-Recipes

The Tratto project fine-tunes the CodeLLaMa language model using the `llama-recipes` repository. All [setup](#setup) instructions should be completed before performing the following steps.

## 1. CodeLLaMa

### 1.1. Download CodeLLaMa

Follow the instructions on the [CodeLLaMa](https://github.com/facebookresearch/codellama) GitHub page to download the model weights for CodeLLaMa.

### 1.2. Convert to HuggingFace

Follow the instructions on the [llama-recipes](https://github.com/facebookresearch/llama-recipes/tree/main?tab=readme-ov-file#model-conversion-to-hugging-face) GItHub page to convert the downloaded CodeLLaMa model into their HuggingFace format.

[//]: # (### 1.3. &#40;Optional&#41; Fine-tune with example dataset)

[//]: # ()
[//]: # (As a sanity check, run the command )

## 2. Data

### 2.1. Add a data configuration scheme

Append the following data classes to the [datasets](https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/configs/datasets.py) file in llama-recipes.

```python
from dataclasses import dataclass

# ... other schema ... 

@dataclass
class oracles_dataset:
    dataset: str = "oracles_dataset"
    file: str = "path/to/TrattoUtil/get_oracles_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    
@dataclass
class tokens_dataset:
    dataset: str = "tokens_dataset"
    file: str = "path/to/TrattoUtil/get_tokens_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
```

### 2.2. Register the dataset

Add keys for the new datasets to the [dataset utils](https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/utils/dataset_utils.py) script.

```python
def get_custom_dataset(dataset_config, tokenizer, split: str):
    # ... implementation ...
    pass


DATASET_PREPROC = {
    # ... other dataset keys ...
    "oracles_dataset": get_custom_dataset,
    "tokens_dataset": get_custom_dataset
}
```

### 2.3. Set dataset field in training

Add the oracles or tokens dataset arguments to the fine-tuning command. An example with the Oracles Dataset is shown below:

```bash
python -m llama_recipes.finetuning \
  --dataset "oracles_dataset" \
  --custom_dataset.file "path/to/TrattoUtil/src/main/data/oracles_dataset.py" 
  [TRAINING_PARAMETERS]
```




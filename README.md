# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

```mermaid
flowchart LR
    A[Receipt images in the selected folder] --> B[Encode each image as a data URL]
    B --> C[DeepSeek vision extraction via LangChain]
    C --> D[Parse summary, discounts, and positive item charges]
    D --> E[Use Decimal to sum final payments]
    D --> F[Cross-check original totals and reread mismatches]
    F --> I[Use Decimal to sum original totals]
    E --> G[Two single-amount HKD answers]
    I --> G
    G --> H[Provided runner writes results.csv]
```

The chain sends each receipt separately to the required DeepSeek vision model through
LangChain in non-thinking mode at temperature zero and asks for structured values
from the transaction portion of the image.
It distinguishes the subtotal from the final payment after rounding, and extracts
each negative promotion, coupon, member, app, percentage, or packaging-damage line
once, without treating rounding as a discount. It also extracts the positive
item charges to cross-check the original total and rereads inconsistent receipts.
The program parses amounts as `Decimal`, sums final payments for the first
question, and uses subtotals plus discount magnitudes, checked against positive
item charges, for the second. It returns exactly one HKD amount for each fixed
query, so the provided runner can write and grade `results.csv`.

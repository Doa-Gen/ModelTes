### Reproduction process
#### deploying lightweight models locally


#### Build the project structure

#### Generate summary samples
	Locally invoke BERT-small and T5-small to read the original text from "read.txt" and generate summaries which are saved in "output/summary_result.txt".
#### Evaluation was conducted using GPT-3.5
##### Invoke the gpt-3.5-turbo model using an api proxy to conduct three evaluations of relevance
###### Zero-shot evaluation
			**Prompt:** Please determine whether the provided summary is consistent with the corresponding article. Note that “consistency" refers to how much information included in the summary is present in the source article.
			**Answer:** (yes or no)
###### Chain of thoughts Evaluation
			**Prompt:** Please determine whether the provided summary is consistent with the corresponding article. Note that “consistency" refers to how much information included in the summary is present in the source article.
			**Answer:** Explain your reasoning step by step then answer the question (yes or no).
###### Score Evaluation
			**Prompt:** Score the following summary given the corresponding article with respect to consistency from 0 to 1 where 1 means most consistent. Note that “consistency" refers to how much information included in the summary is present in the source article.
			**Score:** (0, 1)

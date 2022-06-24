# Theory-Grounded Measurement of U.S. Social Stereotypes in English Language Models

By: [Yang Trista Cao]`<ycao95@umd.edu>`, [Anna Sotnikova] `<asotniko@umd.edu>`, [Hal Daumé III](http://hal3.name) `<hal3@umd.edu>`, [Rachel Rudinger]`<rudinger@umd.edu>` and [Linda Zou]`<lxzou@umd.edu>`

As part of a study of stereotypes in English Language Models, we collected and annotated a dataset of people’s perceptions on different social groups in the United States. We call this dataset the **Crowdsourced U.S. Stereotypes Measured across Agency, Beliefs, and Communion**; what follows below is the [datasheet](https://arxiv.org/abs/1803.09010) describing this data. If you use this dataset, please acknowledge it by citing [the original paper](https://arxiv.org/abs/2206.11684):

```
@inproceedings{}
}
```


## Motivation


1. **For what purpose was the dataset created?** *(Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.)*
    
    This dataset was created to study stereotypes in masked English language models and measure how stereotypes in models align with human ones. The approach for measuring stereotypes is built on the Agency Beliefs Communion (ABC) model, which measures stereotypes toward a social group with respect to 16 traits in three dimensions: Agency/Socioeconomic Success, Conservative–Progressive Beliefs, and Communion.

1. **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**
    
    This dataset was created by Yang Trista Cao, Anna Sotnikova, Hal Daumé III, Rachel Rudinger, and Linda Zou. At the time of creation, Cao and Sotnikova were graduate students at the University of Maryland (UMD), and Daumé, Rudinger, and Zou are Professors there.


1. **Who funded the creation of the dataset?** *(If there is an associated grant, please provide the name of the grantor and the grant name and number.)*
    
    Funding for Cao and Daum\'e III was provided in part from the National Science Foundation project 2131508 and in part from University of Maryland foundation funds. Funding for Sotnikova was provided through the UMD AMSC program. Annotations were crowdsourced and funded through project 2131508.(The annotation was approved by the UMD IRB.)


1. **Any other comments?**
    
    None.





## Composition


1. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** *(Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.)*
    
    The dataset contains three files: aggregated scores, annotators demographics, and full data. 

In the aggregated scores file, each social group has scores for all 16 trait pairs. For more details on groups and trait pair, please, see the paper. Each score is an average across all annotations per trait pair per group.

In the demographic data, we have demographic information such as gender, age, education, race, state where the respondent currently lives, number of years lived in that states, and the territory type (urban or rural). In total, we have demographic information for 133 annotators. Please, note that annotators IDs were assigned at random and are not supposed to be used for getting relations between the annotators answers and their demographics.


In full data file, we have 133 annotators and theirs annotations for 4 social groups. 4 social groups were randomly sampled from the list of 26 social groups until every group got 20 annotations. For each social groups there are scores for 16 trait pairs. Scores are from 0 to 100, where 0 means that left trait is applicable, while 100 means that the right trait is applicable to the social group. Score of 50 means neutral. For instance score  of 10 for the pair poor-wealthy for some social group would mean that this social group vowed as poor.


2. **How many instances are there in total (of each type, if appropriate)?**
    
    In total, the dataset contains annotations from 133 annotators. Consequently, we have 133 annotators in full data file, and 133 demographic records in annotators demographics file. In the aggregated scores, we have scores for all 16 trait pairs for all 26 single groups.

3. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** *(If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).)*
    
    Our dataset contains all instances and its aggregations that passed the annotation quality check (for the details, please, see the paper above).


4. **What data does each instance consist of?** *(``Raw'' data (e.g., unprocessed text or images)or features? In either case, please provide a description.)*
    
    In case of the aggregated data, each instance contains scores per trait pair per social group. In case of the full data, each instance represents one annotator and his/her scores for each of 16 trait pairs for four annotated groups. For the demographic data, this is the information provided by the annotators. Please, note that some annotators may not provide some information about themselves.


5. **Is there a label or target associated with each instance? If so, please provide a description.**
    
    In the full data file, all instances are identified by annotator’s IDs.
In the aggregated data file, each instance is presented by aggregated scores for 16 trait pairs (please, see details in the paper).
In the demographic data, as in the full data instance is identified by annotator’s IDs. However, these ID do not correspond to the full data IDs. This is done in order to avoid potential harms of misinterpretation of scores based on annotator’s demographics.

6. **Is any information missing from individual instances?** *(If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.)*
    
    No.


7. **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** *( If so, please describe how these relationships are made explicit.)*
    
  No. Crowd workers did the annotation task independently. Each crowd worker annotated four social groups.

8. **Are there recommended data splits (e.g., training, development/validation, testing)?** *(If so, please provide a description of these splits, explaining the rationale behind them.)*
    
    We expect this data to be used for testing purposes.

    We do not explicitly provide a training/validation/testing split;   

9. **Are there any errors, sources of noise, or redundancies in the dataset?** *(If so, please provide a description.)*
    
   There are no errors due to data processing. However, one should always keep in mind that all annotations are subjective. 


10. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *(If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.)*
    
    The dataset is self-contained.


11. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** *(If so, please provide a description.)*
    
    No; All data was collected from crowd workers through Prolific platform, any information that might identify them is not released.

12. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** *(If so, please describe why.)*
    
   We present scores for 16 trait pairs for a variety of social groups, these scores might present certain groups not in the complimentary way. This potentially might be harmful to representatives of these social groups.

13. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
    
    Yes, all scores are given to traits describing certain social groups.

14. **Does the dataset identify any subpopulations (e.g., by age, gender)?** *(If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.)*
    
   We collect annotations for 26 social groups, each social group has 20 annotations from different crowd workers.

15. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** *(If so, please describe how.)*
    
    No.


16. **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** *(If so, please provide a description.)*
    
    Scores for trait pairs associated with social groups might negatively affect these groups.


17. **Any other comments?**
    
    None.





## Collection Process


1. **How was the data associated with each instance acquired?** *(Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.)*
    
    The data was collected through Prolific platform. Crowd workers filled out the survey. In order to do the survey, participants first had to read and agree with the provided consent form, which was approved by UMD IRB office.

1. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *(How were these mechanisms or procedures validated?)*
    
   Prolific was used to post tasks for annotators. Once annotators sing up to do the task, they are forwarded to Qualtrics survey page. Qualtrics was used to create the survey. 

1. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**
    
   No.

1. **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**
    
    The survey was completed by crowd workers, who were paid $12.00 per hour. Maryland’s current minimum wage is $12.20

1. **Over what timeframe was the data collected?** *(Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?  If not, please describe the timeframe in which the data associated with the instances was created.)*
    
    The dataset was collected in the Fall of 2021.

1. **Were any ethical review processes conducted (e.g., by an institutional review board)?** *(If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.)*
    
    Data collection process was approved by the University of Maryland IRB. Title: [1724519-1] Assessing and measuring multilingual and cross-culture stereotypes in language technology systems.

1. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
    
    Yes; 


1. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**
    
   Directly, all individuals filled the provided survey.

1. **Were the individuals in question notified about the data collection?** *(If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.)*
    
    Yes, they were provided with the consent form in the beginning of the survey.

1. **Did the individuals in question consent to the collection and use of their data?** *(If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.)*
    
   Yes. Please, see the consent form in the previous question.

1. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** *(If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).)*
    
    Yes, they have an option to contact authors with such request.

1. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** *(If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.)*
    
    Yes, please, see the related paper. 


1. **Any other comments?**
    
    None.





## Preprocessing/cleaning/labeling


1. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *(If so, please provide a description. If not, you may skip the remainder of the questions in this section.)*
    
    Yes; In data with annotators demographic, we removed annotators who did not provide either their gender or age. In addition, we removed annotator’s ids for demographics data so that annotator’s demographic cannot be associated with his/her/their responses. This will help us to avoid potential harms to representatives of the same social groups as annotators.

1. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *(If so, please provide a link or other access point to the "raw" data.)*
    
   No.

1. **Is the software used to preprocess/clean/label the instances available?** *(If so, please provide a link or other access point.)*
    
    No.


1. **Any other comments?**
    
    None.





## Uses


1. **Has the dataset been used for any tasks already?** *(If so, please provide a description.)*
    
    The dataset has been used to study stereotypical associations in English language models.

1. **Is there a repository that links to any or all papers or systems that use the dataset?** *(If so, please provide a link or other access point.)*
    
    No.


1. **What (other) tasks could the dataset be used for?**
    
    The dataset could be used for studying stereotypes in language models. 


1. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** *(For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks)  If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?)*
    
    
    This dataset has a few limitations that should be considered before usage. First, the dataset is limited to English and U.S. social stereotypes. Second, our results are likely affected by reporting bias and by a defaulting effect where, when people say men they may actually mean cis straight white men, because the defaults go unremarked.
In addition, the collected data may be affected by social desirability bias, which refers to people's tendency to respond in socially acceptable ways, though all the annotation activities are anonymous. Third, asking survey questions also raises the possibility of this work inadvertently reinforcing stereotypes in participants.


1. **Are there tasks for which the dataset should not be used?** *(If so, please provide a description.)*
    
    This dataset should not be used for any systems which are deployed for real-users in tasks such as classification, masked words fillings etc. This is a research purpose only dataset.

2. **Any other comments?**
    
    None.




## Distribution


1. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *(If so, please provide a description.)*
    
    Yes, the dataset is freely available.


1. **How will the dataset will be distributed (e.g., tarball  on website, API, GitHub)?** *(Does the dataset have a digital object identifier (DOI)?)*
    
    The dataset is free for download at https://github.com/TristaCao/U.S_Stereotypes.


1. **When will the dataset be distributed?**
    
    The dataset is distributed as of May 2022 in its first version.


1. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *(If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.)*
    
    The dataset is licensed under a MIT license.


1. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.)*
    
    Not to our knowledge.


1. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.)*
    
    Not to our knowledge.


1. **Any other comments?**
    
    None.





## Maintenance


1. **Who is supporting/hosting/maintaining the dataset?**
    
    Trista Cao and Anna Sotnikova are maintaining. Cao is hosting on github.


1. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
    
    E-mail addresses are at the top of this document.


1. **Is there an erratum?** *(If so, please provide a link or other access point.)*
    
    Currently, no. As errors are encountered, future versions of the dataset may be released (but will be versioned). They will all be provided in the same github location.


1. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances')?** *(If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?)*
    
    Same as previous.


1. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** *(If so, please describe these limits and explain how they will be enforced.)*
    
    No.


1. **Will older versions of the dataset continue to be supported/hosted/maintained?** *(If so, please describe how. If not, please describe how its obsolescence will be communicated to users.)*
    
    Yes; all data will be versioned.


1. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *(If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.)*
    
    Errors may be submitted via the bugtracker on github. More extensive augmentations may be accepted at the authors' discretion.


1. **Any other comments?**
    
    None.



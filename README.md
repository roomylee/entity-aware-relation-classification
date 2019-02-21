# Entity-aware Attention for Relation Classification

![model](https://user-images.githubusercontent.com/15166794/52579582-c7339100-2e69-11e9-9081-711e7576e717.png)

This repository contains the official TensorFlow implementation of the following paper:

> **Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing**<br>
> Joohong Lee, Sangwoo Seo, Yong Suk Choi<br>
> [https://arxiv.org/abs/1901.08163](https://arxiv.org/abs/1901.08163)
> 
> **Abstract:** *Classifying semantic relations between entity pairs in sentences is an important task in Natural Language Processing (NLP). Most previous models for relation classification rely on the high-level lexical and syntactic features obtained by NLP tools such as WordNet, dependency parser, part-of-speech (POS) tagger, and named entity recognizers (NER). In addition, state-of-the-art neural models based on attention mechanisms do not fully utilize information of entity that may be the most crucial features for relation classification. To address these issues, we propose a novel end-to-end recurrent neural model which incorporates an entity-aware attention mechanism with a latent entity typing (LET) method. Our model not only utilizes entities and their latent types as features effectively but also is more interpretable by visualizing attention mechanisms applied to our model and results of LET. Experimental results on the SemEval-2010 Task 8, one of the most popular relation classification task, demonstrate that our model outperforms existing state-of-the-art models without any high-level features.*

## Usage
### Train / Test
* Train data is located in "*<U>SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT*</U>".
* You can apply some pre-trained word embeddings: [word2vec](https://code.google.com/archive/p/word2vec/), [glove100](https://nlp.stanford.edu/projects/glove/), [glove300](https://nlp.stanford.edu/projects/glove/), and [elmo](https://tfhub.dev/google/elmo/1). The pre-trained files should be located in `resource/`. [Check this code](https://github.com/roomylee/entity-aware-relation-classification/blob/f77668088210ce2bb0e94033bdf1cabb45c0bbf0/train.py#L115).
* In every evaluation step, the test performance is evaluated by test dataset located in "*<U>SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT*</U>".

##### Display help message:
```bash
$ python train.py --help
```
##### Train Example:
```bash
$ python train.py --embeddings glove300
```


## Visualization
* Self Attention
![sa](https://user-images.githubusercontent.com/15166794/52579583-c7339100-2e69-11e9-93a3-b1aa2aafa19f.png)
* Latent Type Representations
![vec](https://user-images.githubusercontent.com/15166794/52579615-d6b2da00-2e69-11e9-80cb-3b566c28918a.png)
* Sets of Entities grouped by Latent Type
![type](https://user-images.githubusercontent.com/15166794/52579616-d74b7080-2e69-11e9-9e3a-c027eb01413b.png)


## SemEval-2010 Task #8
* Given: a pair of *nominals*
* Goal: recognize the semantic relation between these nominals.
* Example:
	* "There were apples, **<U>pears</U>** and oranges in the **<U>bowl</U>**." 
		<br> → *CONTENT-CONTAINER(pears, bowl)*
	* “The cup contained **<U>tea</U>** from dried **<U>ginseng</U>**.” 
		<br> → *ENTITY-ORIGIN(tea, ginseng)*


### The Inventory of Semantic Relations
1. *Cause-Effect(CE)*: An event or object leads to an effect(those cancers were caused by radiation exposures)
2. *Instrument-Agency(IA)*: An agent uses an instrument(phone operator)
3. *Product-Producer(PP)*: A producer causes a product to exist (a factory manufactures suits)
4. *Content-Container(CC)*: An object is physically stored in a delineated area of space (a bottle full of honey was weighed) Hendrickx, Kim, Kozareva, Nakov, O S´ eaghdha, Pad ´ o,´ Pennacchiotti, Romano, Szpakowicz Task Overview Data Creation Competition Results and Discussion The Inventory of Semantic Relations (III)
5. *Entity-Origin(EO)*: An entity is coming or is derived from an origin, e.g., position or material (letters from foreign countries)
6. *Entity-Destination(ED)*: An entity is moving towards a destination (the boy went to bed) 
7. *Component-Whole(CW)*: An object is a component of a larger whole (my apartment has a large kitchen)
8. *Member-Collection(MC)*: A member forms a nonfunctional part of a collection (there are many trees in the forest)
9. *Message-Topic(CT)*: An act of communication, written or spoken, is about a topic (the lecture was about semantics)
10. *OTHER*: If none of the above nine relations appears to be suitable.


### Distribution for Dataset
* **SemEval-2010 Task #8 Dataset [[Download](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?layout=list&ddrp=1&sort=name&num=50#)]**

	| Relation           | Train Data          | Test Data           | Total Data           |
	|--------------------|:-------------------:|:-------------------:|:--------------------:|
	| Cause-Effect       | 1,003 (12.54%)      | 328 (12.07%)        | 1331 (12.42%)        |
	| Instrument-Agency  | 504 (6.30%)         | 156 (5.74%)         | 660 (6.16%)          |
	| Product-Producer   | 717 (8.96%)         | 231 (8.50%)         | 948 (8.85%)          |
	| Content-Container  | 540 (6.75%)         | 192 (7.07%)         | 732 (6.83%)          |
	| Entity-Origin      | 716 (8.95%)         | 258 (9.50%)         | 974 (9.09%)          |
	| Entity-Destination | 845 (10.56%)        | 292 (10.75%)        | 1137 (10.61%)        |
	| Component-Whole    | 941 (11.76%)        | 312 (11.48%)        | 1253 (11.69%)        |
	| Member-Collection  | 690 (8.63%)         | 233 (8.58%)         | 923 (8.61%)          |
	| Message-Topic      | 634 (7.92%)         | 261 (9.61%)         | 895 (8.35%)          |
	| Other              | 1,410 (17.63%)      | 454 (16.71%)        | 1864 (17.39%)        |
	| **Total**          | **8,000 (100.00%)** | **2,717 (100.00%)** | **10,717 (100.00%)** |



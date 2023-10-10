# Business-Buddy

## Description
Business Buddy is a data analytics tool that leverages NLP techniques to obtain business insights from text data. It automatically generates and labels topics within the text data and will give you the sentiment surrounding those topics

## Data Input
It is currently implemented with pandas dataframes however any text data inside of a list can be passed into the function.

## Example
Say we have reviews for a burger restaurant. Topics such as Burger, Service, Fries, etc will be generated. In addition to these automatically generated labeled topics, we will get the corresponding sentiments associated with the topics. So in this case a lot of people might review good burger or have a positive sentiment towards the burger. On the flipside, we could have bad sentiment associated with the fries. In this case it is important to have the associated sentiment to fully understand what and how people feel in relation to your business.

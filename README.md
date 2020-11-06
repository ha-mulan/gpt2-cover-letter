# gpt2-cover-letter

generate cover-letter sentence based fine-tuned gpt2 model

# How to use
This API has 2 routing address

# POST parameter

/long

text : begining of the text you want to generate

number_samples : the number of sentence that will be generated

length : the length of each sentence

/short

text : begining of the text you want to generate

number_samples : the number of word that will be generated

# With CLI :

curl --location --request POST 'https://main-gpt2-cover-letter-jeong-hyun-su.endpoint.ainize.ai/gpt2-cover-letter/short' --form 'text=We have more cases' --form 'num_samples=5'

curl --location --request POST 'https://main-gpt2-cover-letter-jeong-hyun-su.endpoint.ainize.ai/gpt2-cover-letter/long' --form 'text=We have more cases' --form 'num_samples=5' --form 'length=10'


# With swagger :

You can test this API with swagger.yaml on swagger editor
# ALBERT PERSIAN LAB
Hi there :wave:, and welcome! In this repository, I'm going to show you how we can use ALBERT-Persian models using HuggingFace in some down-stream NLP tasks. You can use this application with/without docker!

[![ALBERT-Persian Demo](/assets/albert-fa-base-v2.png)](https://youtu.be/QmoLTk0rh8U)

[ALBERT-Persian Playground](http://albert-lab.m3hrdadfi.me/)

## How to run?

### Without Docker:

1. Go to your terminal and clone the project, and install all python packages requirements using:

``` bash
$ git clone https://github.com/m3hrdadfi/albert-persian-lab.git
$ cd albert-persian-lab
$ pip install -r requirements.txt
```

2. Run the Streamlit application using:
``` bash

$ streamlit run app.py
```

### With Docker:

For local usage:

``` bash
$ docker-compose -f local.yml build
$ docker-compose -f local.yml up 
```

For production usage:

``` bash
$ docker-compose -f product.yml build
$ docker-compose -f product.yml up 
```

## License

ALBERT Persian is completely free and open source and licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

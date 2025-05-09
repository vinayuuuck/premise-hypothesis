# Note
When cloning for the first time, perform the following steps
to ensure all the required data is present.

## 1. Download SNLI data from the following link:
https://nlp.stanford.edu/projects/snli/

## 2. Create a directory named `data` in the root of the project:

```
mkdir data
```

## 3. Move the downloaded SNLI data into the `data` directory:

From the command line, run the following command:
```
mv snli_1.0.zip data/
```
or just use the file explorer to move the file,
and then unzip it:
```
unzip data/snli_1.0.zip
```

## 4. Download the GloVe embeddings from the following link:
https://nlp.stanford.edu/data/glove.6B.zip

## 5. Move the downloaded GloVe embeddings into the `data` directory:
From the command line, run the following command:
```
mv glove.6B.zip data/
```
or just use the file explorer to move the file,
and then unzip it:
```
unzip data/glove.6B.zip
```

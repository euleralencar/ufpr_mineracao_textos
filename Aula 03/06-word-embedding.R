## ---- message = FALSE---------------------------------------------------------
#-----------------------------------------------------------------------
# Pacotes.

library(jsonlite)  # Ler e escrever para JSON.
library(tidyverse) # Recursos para manipulação e visualização de dados.
library(tidytext)  # Manipulação de texto a la tidyverse.
library(text2vec)  # Para medidas de distância e Glove.
library(tm)        # Recursos para mineração de texto.
library(wordcloud) # Nuvem de palavras.


## ---- cache = TRUE------------------------------------------------------------
#-----------------------------------------------------------------------
# Importação do texto.

# Endereço de arquivo JSON com avaliação de veículos.
url <- paste0("https://github.com/leg-ufpr/hackathon/blob/master",
              "/opinioes.json?raw=true")

# Importa reviews de veículos.
txt <- fromJSON(url)
str(txt)


## -----------------------------------------------------------------------------
# Conteúdo está na forma de matriz.
# txt[1, ]

# Passando para tabela.
colnames(txt) <- c("id", "title", "model", "owner", "condition", "good",
                   "bad", "defect", "general", "ts")
tt <- as_tibble(txt)
glimpse(tt)

# Modelos de veículos contidos nas avaliações.
tt$product <- tt$model %>%
    str_extract("^([[:alpha:]]+ +[[:alpha:]]+)") %>%
    str_to_upper()

# Tipos únicos.
# tt$product %>% unique() %>% dput()
tt %>%
    count(product, sort = TRUE)

# Aplica filtro para reter apenas um modelo de carro.
mod <- c("CHEVROLET CELTA",
         "CHEVROLET ONIX",
         "FIAT PALIO",
         "FIAT UNO",
         "HYUNDAI HB",
         "RENAULT SANDERO",
         "VOLKSWAGEN FOX",
         "VOLKSWAGEN GOL")[7]
texto <- tt %>%
    filter(str_detect(product, mod)) %>%
    select(id, general)
texto


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Cria o corpus a partir de um vetor.

preprocess <- function(x) {
    x <- tolower(x)
    x <- gsub(pattern = "[[:punct:]]+", replacement = " ", x = x)
    x <- removeWords(x, words = stopwords("portuguese"))
    x <- removeWords(x, words = c("opinião", "geral", "carro", "veículo"))
    x <- removeNumbers(x)
    x <- gsub(pattern = "[[:space:]]+", replacement = " ", x = x)
    x <- iconv(x, to = "ASCII//TRANSLIT")
    x <- trimws(x)
    return(x)
}

# Aplica o preprocessamento.
system.time({
    xx <- preprocess(texto$general)
    xx <- xx[nchar(xx) >= 50]
})


## ---- eval = FALSE, include = FALSE-------------------------------------------
## # system.time({
## #     cps <- VCorpus(VectorSource(texto$general)) %>%
## #         tm_map(FUN = content_transformer(tolower)) %>%
## #         tm_map(FUN = content_transformer(
## #                    function(x) gsub("[[:punct:]]", " ", x))) %>%
## #         tm_map(FUN = removeWords,
## #                words = stopwords("portuguese")) %>%
## #         tm_map(FUN = removeWords,
## #                words = c("opinião", "geral", "carro", "veículo")) %>%
## #         tm_map(FUN = removeNumbers) %>%
## #         tm_map(FUN = content_transformer(
## #                    function(x) gsub("[[:space:]]+", " ", x))) %>%
## #         tm_map(FUN = content_transformer(
## #                    function(x) iconv(x, to = "ASCII//TRANSLIT"))) %>%
## #         tm_map(FUN = trimws)
## #     cps <- tm_filter(cps,
## #                      FUN = function(x) {
## #                          # sum(nchar(content(x))) >= 50
## #                          sum(nchar(x)) >= 50
## #                      })
## #     xx <- unlist(content(cps))
## # })


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Tokenização e criação do vocabulário.

# Faz a tokenização de cada documento.
tokens <- space_tokenizer(xx)
str(tokens, list.len = 4)

# Cria iterador.
iter <- itoken(tokens)
class(iter)

# Cria o vocabulário em seguida elimina palavras de baixa ocorrência.
vocab <- create_vocabulary(iter, ngram = c(1, 1))
vocab <- prune_vocabulary(vocab, term_count_min = 3)
str(vocab)

# Cria a matriz de co-ocorrência de termos (term cooccurrence matrix).
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it = iter,
                  vectorizer = vectorizer,
                  skip_grams_window = 5)
str(tcm)

# Esparsidade da TCM.
1 - length(tcm@x)/prod(tcm@Dim)


## ---- results = "hide"--------------------------------------------------------
#-----------------------------------------------------------------------
# Aplica o GloVe.

# QUESTION: qual o tamanho de vetor usar?

args(GlobalVectors$new)

# Inicializa objeto da classe.
glove <- GlobalVectors$new(rank = 30,
                           x_max = 10, )
names(glove)

# Ajuste a rede neuronal e determinação dos word vectors.
# wv_main <- glove$fit_transform(tcm, n_iter = 90)
wv_main <- glove$fit_transform(tcm,
                               n_iter = 50,
                               convergence_tol = 0.01,
                               n_threads = 8)
dim(wv_main)

wv_context <- glove$components
dim(wv_context)

word_vectors <- wv_main + t(wv_context)
dim(word_vectors)


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Uso dos vetores para medidas de similaridade.

# Termo alvo.
tgt <- rbind(word_vectors["consumo", ])

# Distância Euclidiana para com os demais termos.
euc <- text2vec::dist2(x = word_vectors, y = tgt)

# head(sort(euc[, 1]), n = 15) %>%
tb_fq <- euc[, 1] %>%
    enframe("Termo", "Distância") %>%
    mutate_at("Distância", round, digits = 5)

ggplot(top_n(tb_fq, Distância, n = 50)) +
    geom_col(mapping = aes(x = reorder(Termo, Distância),
                           y = Distância)) +
    coord_flip()

# Similaridade do coseno (convertida para distância em seguida).
sim  <- sim2(x = word_vectors, y = tgt, method = "cosine")
sim <- (1 - sim)/2

tb_fq <- sim[, 1] %>%
    enframe("Termo", "Distância") %>%
    mutate_at("Distância", round, digits = 5)

ggplot(top_n(tb_fq, Distância, n = 50)) +
    geom_col(mapping = aes(x = reorder(Termo, Distância),
                           y = Distância)) +
    coord_flip()


## ---- message = FALSE, cache = TRUE-------------------------------------------
#-----------------------------------------------------------------------
# Leitura e preprocessamento dos dados.

# Dataset com mais de 70 mil registros.
url <- paste0("http://leg.ufpr.br/~walmes/data",
              "/TCC_Brasil_Neto/ImoveisWeb-Realty.csv")
tb <- read_csv2(url, locale = locale(encoding = "latin1"))
str(tb, give.attr = FALSE, vec.len = 1)

# Preprocessamento da descrição dos imóveis.
system.time({
    xx <- tb$description
    xx <- xx[nchar(xx) >= 80]
    xx <- preprocess(xx)
})


## ---- eval = FALSE, include = FALSE-------------------------------------------
## system.time({
##     cps <- VCorpus(VectorSource(tb$description)) %>%
##         tm_map(FUN = content_transformer(tolower)) %>%
##         tm_map(FUN = content_transformer(
##                    function(x) gsub("[[:punct:]]", " ", x))) %>%
##         tm_map(FUN = removeWords,
##                words = stopwords("portuguese")) %>%
##         tm_map(FUN = removeWords,
##                words = c("opinião", "geral", "carro", "veículo")) %>%
##         tm_map(FUN = removeNumbers) %>%
##         tm_map(FUN = content_transformer(
##                    function(x) gsub("[[:space:]]+", " ", x))) %>%
##         tm_map(FUN = content_transformer(
##                    function(x) iconv(x, to = "ASCII//TRANSLIT"))) %>%
##         tm_map(FUN = trimws)
##     cps <- tm_filter(cps,
##                      FUN = function(x) {
##                          # sum(nchar(content(x))) >= 50
##                          sum(nchar(x)) >= 50
##                      })
##     xx <- unlist(content(cps))
## })


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Tokenização e criação da TCM.

# Faz a tokenização de cada documento.
tokens <- space_tokenizer(xx)
str(tokens, list.len = 4)

# Cria iterador.
iter <- itoken(tokens)
class(iter)

# Cria o vocabulário em seguida elimina palavras de baixa ocorrência.
vocab <- create_vocabulary(iter, ngram = c(1, 1))
vocab <- prune_vocabulary(vocab, term_count_min = 3)
str(vocab)

# Cria a matriz de co-ocorrência de termos.
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it = iter,
                  vectorizer = vectorizer,
                  skip_grams_window = 5)
str(tcm)

# Esparsidade da TCM.
1 - length(tcm@x)/prod(tcm@Dim)


## ---- results = "hide"--------------------------------------------------------
#-----------------------------------------------------------------------
# Ajuste do GloVe.

# O GloVe está implementado como programação orientada a objeto na
# arquitetura R6. O pacote R6 foi carregado.
s <- sessionInfo()
"R6" %in% union(names(s$otherPkgs), names(s$loadedOnly))

# Inicializa o objeto.
glove <- GlobalVectors$new(rank = 50,
                           x_max = 10)

# Ajuste a rede neuronal e determinação dos word vectors.
# wv_main <- glove$fit_transform(tcm, n_iter = 25)
wv_main <- glove$fit_transform(tcm, n_iter = 50)
wv_context <- glove$components
word_vectors <- wv_main + t(wv_context)


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Aplicação do GloVe.

nearest_terms <- function(word_vec, word_vectors, n = 10) {
    tgt <- rbind(word_vec)
    euc <- text2vec::dist2(x = word_vectors, y = tgt)
    head(sort(euc[, 1]), n = n) %>%
        enframe("Termo", "Distância") %>%
        mutate_at("Distância", round, digits = 5)
}

nearest_terms(word_vectors["quarto", ], word_vectors, n = 10)
nearest_terms(word_vectors["banheiro", ], word_vectors, n = 10)
nearest_terms(word_vectors["cozinha", ], word_vectors, n = 10)

# Álgebra.
a <- word_vectors["dormitorios", ] -
    word_vectors["dormitorio", ] +
    word_vectors["quarto", ]
a

nearest_terms(a, word_vectors, n = 4)


## ---- message = FALSE---------------------------------------------------------
#-----------------------------------------------------------------------
# Carrega dados de binário.

# Carrega os dados.
load("./ufpr-news.RData")
length(ufpr)
str(ufpr[[1]])

# Extrai títulos das notícias.
tit <- sapply(ufpr, "[[", "str_titulo")
dul <- duplicated(tolower(tit))
sum(dul)

# Removendo as duplicações com base nos títulos.
ufpr <- ufpr[!dul]

# Extrai o conteúdo das notícias.
x <- sapply(ufpr, FUN = "[", "conteudo_texto")
x <- unlist(x)

# Mostra o conteúdo dos primeiros documentos.
head(x, n = 2) %>%
    map(str_sub, start = 1, end = 500) %>%
    map(str_wrap, width = 60) %>%
    walk(cat, "... <continua> ... \n\n")

#-----------------------------------------------------------------------
# Preprocessamento, faz a tokenização e criação da TCM.

# Aplica o preprocessamento.
xx <- x
xx <- xx[nchar(xx) >= 100]
xx <- preprocess(xx)

# Faz a tokenização de cada documento.
tokens <- space_tokenizer(xx)
str(tokens, list.len = 4)

# Cria iterador.
iter <- itoken(tokens)
class(iter)

# Cria o vocabulário em seguida elimina palavras de baixa ocorrência.
# vocab <- create_vocabulary(iter, ngram = c(1, 2)) # Bigramas.
vocab <- create_vocabulary(iter, ngram = c(1, 1))
vocab <- prune_vocabulary(vocab, term_count_min = 3)
str(vocab)

# Cria a matriz de co-ocorrência de termos.
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it = iter,
                  vectorizer = vectorizer,
                  skip_grams_window = 5)
str(tcm)

# Esparsidade da TCM.
1 - length(tcm@x)/prod(tcm@Dim)


## ---- results = "hide"--------------------------------------------------------
#-----------------------------------------------------------------------
# Ajuta o GloVe.

# Inicializa objeto (arquitetura de POO R6).
glove <- GlobalVectors$new(rank = 50,
                           x_max = 10)

# Ajuste a rede neuronal e determinação dos word vectors.
# wv_main <- glove$fit_transform(tcm, n_iter = 25)
wv_main <- glove$fit_transform(tcm, n_iter = 25)
wv_context <- glove$components
word_vectors <- wv_main + t(wv_context)


## ---- warning = FALSE---------------------------------------------------------
#-----------------------------------------------------------------------
# Núvem de palavras.

tgt <- rbind(word_vectors["vestibular", ])
sim  <- sim2(x = word_vectors, y = tgt, method = "cosine")
freq <- head(sort(sim[, 1], decreasing = TRUE), n = 300)
cbind(head(freq, n = 10))

# Termos mais similares.
oldpar <- par()
par(mar = c(0, 0, 0, 0))
wordcloud(words = names(freq),
          freq = freq^2,
          random.order = FALSE,
          rot.per = 0,
          colors = tail(brewer.pal(9, "Blues"), n = 5))
par(oldpar)

# Gráfico de barras.
freq %>%
    enframe() %>%
    top_n(value, n = 50) %>%
    ggplot() +
    geom_col(mapping = aes(x = reorder(name, value), y = value)) +
    coord_flip()

# tgt <- rbind(word_vectors["estatistica", ])
# tgt <- rbind(word_vectors["professor", ])
tgt <- rbind(word_vectors["ufpr", ])
sim  <- sim2(x = word_vectors, y = tgt, method = "cosine")
freq <- head(sort(sim[, 1], decreasing = TRUE), n = 300)
cbind(head(freq, n = 10))

# Termos mais similares.
oldpar <- par()
par(mar = c(0, 0, 0, 0))
wordcloud(words = names(freq),
          freq = freq^2,
          random.order = FALSE,
          rot.per = 0,
          colors = tail(brewer.pal(9, "Reds"), n = 5))
par(oldpar)


## ---- message=FALSE, error=FALSE, warning=FALSE-------------------------------
#-----------------------------------------------------------------------
# Versões dos pacotes e data do documento.

devtools::session_info()
Sys.time()


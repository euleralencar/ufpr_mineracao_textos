# Importação de pacotes ----
library(tm)
library(tidyverse)


# Importação do texto ----
docs <- c("A vida é linda",
          "A vida é uma aventura",
          "A vida é uma só",
          "A vida é linda por ser única",
          "A vida é minha, minha vida!",
          "A vida é pra ser vivida")

# Vamos criar o corpus ----
cps <- VCorpus(VectorSource(x = docs),
               readerControl = list(language = "pt",
                                    load = TRUE))

# Vamos fazer as primeiras transformações ----

cps <- 
cps %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(removeWords,
         words = c("a","é","e","uma","pra",
                   "por","ser","ter"))

content(cps[[1]])

# Vamos aplicar uma função de ponderação ----

apropos("^weight[[:upper:]]", ignore.case = FALSE)

# Ponderação TF ----

dtm_tf <- DocumentTermMatrix(cps,
                             control = list(
                               # usando ponderação TFIDF
                               weighting = weightTf,
                               # tamanho do token de 1 a infinito
                               wordLengths = c(1, Inf)
                             ))

inspect(dtm_tf)

dtm_bin <- 1 * as.matrix(dtm_tf > 0)

# TFIDF manualmente ----

dtm_tf <- as.matrix(dtm_tf)

dtm_bun <- 1 * (dtm_tf > 0)

(idf <- log2((nrow(dtm_bin))/colSums(dtm_bin)))

sweep(dtm_tf, 
      MARGIN = 2, 
      STATS = idf, 
      FUN = "*")

# TF-DF com pacotes ----
# Observe que a variável vida não tem informação

# Agora fazendo com a função do pacote tm
weightTfIdf_un <- function(x) weightTfIdf(x, normalize = FALSE)

dtm_tfidf <- DocumentTermMatrix(cps,
                                control = list(
                                  weighting = weightTfIdf,
                                  wordLenghts = c(1, Inf)
                                ))

inspect(dtm_tfidf)

## ---- message = FALSE---------------------------------------------------------
#-----------------------------------------------------------------------
# Pacotes.

library(jsonlite)    # Leitura e escrita JSON.
library(tidyverse)   # Recursos de manipulação e visualização.
library(tidytext)    # Manipulação de texto a la tidyverse.
library(tm)          # Mineração de texto.
library(topicmodels) # Modelagem de tópicos.
library(wordcloud)   # Núvem de palavras.
library(ggtern)      # Gráfico ternário.

# library(lda)
# library(LDAvis)


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Carrega notícias sobre a UFPR.

# Dados armazenados na forma de lista em binário `RData`.
load("./ufpr-news.RData")
length(ufpr)
str(ufpr[[1]])

#-----------------------------------------------------------------------
# Título das notícias.

# Extrai os títulos.
tit <- sapply(ufpr, "[[", "str_titulo")
# tit <- sapply(ufpr, "[[", "conteudo_texto")
dul <- duplicated(tolower(tit))
sum(dul)

# Removendo as duplicações com base nos títulos.
ufpr <- ufpr[!dul]
tit <- tit[!dul]

#-----------------------------------------------------------------------
# Período das publicações.

dts <- strptime(sapply(ufpr, "[[", "ts_publicacao"),
                format = "%Y-%m-%d %H:%M:%S")
range(dts)

#-----------------------------------------------------------------------
# Veículos de divulgação.

vei <- sapply(ufpr, "[[", "str_veiculo")
tb <- sort(table(vei), decreasing = TRUE)

ggplot(enframe(head(tb, n = 30)),
       aes(x = reorder(name, value), y = value)) +
    geom_col() +
    labs(x = "Veículo", y = "Frequência") +
    coord_flip()

#-----------------------------------------------------------------------
# Extraindo o conteúdo das notícias.

L <- sapply(ufpr, FUN = "[", "conteudo_texto")
L <- unlist(L)

L[1:3] %>%
    map(str_sub, start = 1, end = 500) %>%
    map(str_wrap, width = 60) %>%
    walk(cat, "... <continua> ... \n\n")

#-----------------------------------------------------------------------
# Cria o corpus a partir da lista.

# is.vector(L)
cps <- VCorpus(VectorSource(x = L),
               readerControl = list(language = "pt"))
cps

# Confere os tamanhos.
length(cps) == length(vei)

#-----------------------------------------------------------------------
# Adiciona os metadados aos documentos do corpus. Eles podem ser úteis
# para aplicar filtros e tarefas por estrato.

# `type = "local"` para usar na `tm_filter()` e `tm_index()`.
meta(cps, type = "local", tag = "veiculo") <- vei
meta(cps, type = "local", tag = "titulo") <- tit
meta(cps, type = "local", tag = "ts") <- as.character(dts)

# Consulta os metadados apenas para verificação.
# meta(cps[[5]])
# meta(cps[[5]], tag = "veiculo")
# meta(cps[[5]], tag = "ts")

# Filtra os documentos usando os metadados.
cps2 <- tm_filter(cps,
                  FUN = function(x) {
                      meta(x)[["veiculo"]] == "Gazeta do Povo"
                  })
length(cps2)


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Processamento.

cps2 <- cps2 %>%
    tm_map(FUN = content_transformer(tolower)) %>%
    tm_map(FUN = content_transformer(
               function(x) gsub(" *-+ *", "-", x))) %>%
    # tm_map(FUN = replacePunctuation) %>%
    tm_map(FUN = content_transformer(
               function(x) gsub("[[:punct:]]", " ", x))) %>%
    tm_map(FUN = removeNumbers) %>%
    tm_map(FUN = removeWords,
           words = stopwords("portuguese")) %>%
    tm_map(FUN = stemDocument,
           language = "portuguese") %>%
    tm_map(FUN = stripWhitespace)

# Para ver os fragmentos dos documentos após o pré-processamento.
sapply(cps2[1:2], content) %>%
    map(str_sub, start = 1, end = 500) %>%
    map(str_wrap, width = 60) %>%
    walk(cat, "... <continua> ... \n\n")

#-----------------------------------------------------------------------
# Criar a matriz de documentos e termos.

# Para fazer modelagem de tópicos, requer ponderação `term-frequency`.
# Ela é a opção default.
dtm <- DocumentTermMatrix(cps2)
dtm

# Número de documentos x tamanho do vocabulário.
dim(dtm)

# Remoção de esparsidade para reduzir dimensão.
rst <- removeSparseTerms(x = dtm, sparse = 0.99)
rst

# Número de documentos x tamanho do vocabulário.
dim(rst)

dtm <- rst


## -----------------------------------------------------------------------------
# Essa função requer ponderação padrão: term frequency.
# k: número de assuntos ou temas.
fit <- LDA(x = dtm, k = 3)
fit

# Classe, métodos e conteúdo (é programação orientada a objetos em
# arquitetura S4).
class(fit)
methods(class = "LDA_VEM")
slotNames(fit)
isS4(fit)

# Termos principais (maior frequência) que são, por default, usados para
# rotular tópicos.
terms(fit)
# get_terms(fit)

# Índice que separa os documentos pelo tópico com maior fração. Esse
# seria o resultado da análise de agrupamento fornecida por essa
# abordagem.
classif <- topics(fit)
head(classif)   # Classificação dos primeiros documentos.
table(classif)  # Distribuição dos documentos nas classes.

# Fração de cada tópico por documento (a soma é 1).
# rowSums(fit@gamma[1:6, ])
head(fit@gamma) %>%
    `colnames<-`(paste0("Tópico", 1:fit@k))

# Fração de cada termo em cada documento (a soma é 1).
round(head(t(exp(fit@beta))), digits = 8) %>%
    `colnames<-`(paste0("Tópico", 1:fit@k)) %>%
    `rownames<-`(paste0("Termo", 1:nrow(.)))


## -----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Distribuição dos tópicos.

# Proporção dos tópicos nos documentos.
topic_coef <- tidy(fit, matrix = "gamma")
head(topic_coef)

# Gráfico da mistura a partir de uma amostra.
aux <- sample_n(topic_coef, size = 150) %>%
    arrange(topic, gamma) %>%
    mutate(document = fct_reorder(document, row_number()))

ggplot(data = aux) +
    aes(x = document,
        y = gamma,
        fill = factor(topic)) +
    geom_col(position = "fill") +
    labs(fill = "Tópico predominante") +
    coord_flip()

# Os mesmos dados mas na forma wide.
topicProbs <- as.data.frame(fit@gamma)
names(topicProbs) <- paste0("T", seq_along(names(topicProbs)))
topicProbs$class <- topics(fit)
names(topicProbs)

# Gráfico composicional para k = 3.
if (fit@k == 3) {
    ggtern(data = topicProbs,
           mapping = aes(x = T1,
                         y = T2,
                         z = T3,
                         color = factor(class))) +
        geom_point(alpha = 0.5) +
        labs(color = "Tópico\npredominante") +
        theme_showarrows()
}

#-----------------------------------------------------------------------
# Distribuição dos tópicos.

# Proporção dos termos nos tópicos.
terms_coef <- tidy(fit, matrix = "beta")
head(terms_coef)

# Os termos mais frequentes pro tópico.
topn_terms <- terms_coef %>%
    group_by(topic) %>%
    top_n(n = 50, wt = beta) %>%
    ungroup()
topn_terms

# ggplot(topn_terms) +
#     aes(x = reorder(term, beta), y = beta) +
#     facet_wrap(facets = ~topic, scales = "free_y", drop = FALSE) +
#     geom_col() +
#     coord_flip()

# Faz os gráficos em separado e retorna em lista.
pp <- topn_terms %>%
    group_by(topic) %>%
    do(plot = {
        ggplot(.) +
            aes(x = reorder(term, beta), y = beta) +
            geom_col() +
            labs(x = "Termos", y = "Frequência") +
            coord_flip()
    })
length(pp$plot)

# Invoca a `grid.arrange()` do pacote `gridExtra`.
do.call(what = gridExtra::grid.arrange,
        args = c(pp$plot, nrow = 1))


## ---- warning = FALSE, fig.height = 10, fig.width = 10------------------------
#-----------------------------------------------------------------------
# Núvem de palavras por tópico.

# Termos mais salientes.
topn_terms <- terms_coef %>%
    group_by(topic) %>%
    top_n(300, beta) %>%
    ungroup()

i <- 0
pal <- c("Reds", "Blues", "Greens", "Purples")[1:fit@k]

oldpar <- par()
par(mfrow = c(2, 2), mar = c(0, 0, 0, 0))
topn_terms %>%
    group_by(topic) %>%
    do(plot = {
        i <<- i + 1
        wordcloud(words = .$term,
                  freq = .$beta,
                  min.freq = 1,
                  max.words = 300,
                  random.order = FALSE,
                  colors = tail(brewer.pal(9, pal[i]), n = 5))
    })
layout(1)
par(oldpar)


## ---- fig.height = 10---------------------------------------------------------
# Pega estampa de tempo.
ts <- sapply(cps2, meta, tag = "ts")
ts <- as.POSIXct(ts)

# Documentos e data de publicação.
doc_ts <- tibble(document = unlist(meta(cps2, "id")),
                 ts = parse_datetime(unlist(meta(cps2, "ts"))))

# Junção.
topic_ts <- inner_join(topic_coef, doc_ts)
topic_ts

gg1 <-
    ggplot(topic_ts) +
    aes(x = ts, y = gamma, color = factor(topic)) +
    geom_point() +
    geom_smooth(se = FALSE, span = 0.45) +
    theme(legend.direction = "horizontal",
          legend.position = "top") +
    labs(x = "Data",
         y = "Fração de cada tópico",
         color = "Tópico")

gg2 <-
    ggplot(topic_ts) +
    aes(x = ts) +
    geom_density(fill = "gray30", alpha = 0.5) +
    labs(y = "Densidade de\ndocumentos",
         x = "Data")

gridExtra::grid.arrange(gg1, gg2, ncol = 1, heights = c(4,1))


## ---- eval = FALSE------------------------------------------------------------
## # Extrai o vetor de palavras.
## v <- content(cps)
## 
## lex <- lexicalize(v)
## str(lex, list.len = 4)
## 
## nTerms(dtm) # Palavras de menos de 2 digitos são excluídas.
## 
## # Frequência das palavras do vocabulário no corpus.
## wc <- word.counts(lex$documents, lex$vocab)
## 
## # Para o ajuste do LDA.
## set.seed(1234)
## k <- 5
## niter <- 40
## alpha <- 0.02
## eta <- 0.02
## 
## fit <- lda.collapsed.gibbs.sampler(documents = lex$documents,
##                                    K = k,
##                                    vocab = lex$vocab,
##                                    num.iterations = niter,
##                                    alpha = alpha,
##                                    eta = eta,
##                                    initial = NULL,
##                                    burnin = 0,
##                                    compute.log.likelihood = TRUE)
## 
## # Para verificar se houve convergência.
## plot(fit$log.likelihoods[1, ])
## 
## # As palavras mais típicas de cada tópico.
## top.topic.words(fit$topics, num.words = 10, by.score = TRUE)
## 
## # Os documentos com maior proporção em cada tópico.
## top.topic.documents(fit$document_sums, num.documents = 3)
## 
## # OBS: o valor de 0.01 somado é para evitar 0 porque isso pode dar
## # problema quando for chamada a função `createJSON` que internamente usa
## # a *Jensen Shannon distance*. Veja a discussão:
## # https://github.com/cpsievert/LDAvis/issues/56.
## 
## # Proporção de cada tópico em cada documento.
## theta <- t(apply(fit$document_sums + 0.01,
##                  MARGIN = 2,
##                  FUN = function(x) x/sum(x)))
## head(theta)
## 
## # Proporção de cada termo em cada tópico.
## phi <- t(apply(fit$topics + 0.01,
##                MARGIN = 1,
##                FUN = function(x) x/sum(x)))
## head(phi[, 1:4])
## 
## json_data <- createJSON(phi = phi,
##                         theta = theta,
##                         doc.length = document.lengths(lex$documents),
##                         vocab = lex$vocab,
##                         term.frequency = as.vector(wc))
## 
## serVis(json = json_data)


## ---- message=FALSE, error=FALSE, warning=FALSE-------------------------------
#-----------------------------------------------------------------------
# Versões dos pacotes e data do documento.

devtools::session_info()
Sys.time()


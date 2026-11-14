# =====================================================================
# funciones.R — núcleo estadístico compartido por el notebook, el app y el paquete
# Proyecto 2: Redes profundas densas (MLP) y modelos mixtos con embeddings
# Maestría en Estadística (Ciencia de Datos e IA) - ESPOCH
# ---------------------------------------------------------------------
# Las redes se ENTRENAN con keras3 (solo en el notebook). Para PREDECIR no se
# necesita TensorFlow: pred_red() hace el paso hacia adelante en R puro con los
# pesos guardados (multiplicaciones de matrices, ReLU y softmax).
# =====================================================================

NIV <- c("Dropout", "Enrolled", "Graduate")
NIV_ES <- c(Dropout = "Deserción", Enrolled = "Matriculado", Graduate = "Graduado")
MODELOS <- c("Línea base", "Regresión logística", "Ridge", "Random Forest", "MLP denso", "Modelo mixto")
COL_MODELO <- c("Línea base" = "#94a3b8", "Regresión logística" = "#334155", "Ridge" = "#0f766e",
                "Random Forest" = "#b45309", "MLP denso" = "#0284c7", "Modelo mixto" = "#7c3aed")
COL_CLASE <- c(Dropout = "#dc2626", Enrolled = "#d97706", Graduate = "#0284c7")

# ---------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------
# Notas semestrales dañadas por el formato de hoja de cálculo ("1.366.666.667" = 13,67):
# se eliminan los puntos y se divide entre 10 hasta quedar en 0-20.
limpiar_nota <- function(x) {
  x <- suppressWarnings(as.numeric(gsub("\\.", "", as.character(x))))
  for (i in 1:10) { k <- which(!is.na(x) & x > 20); if (!length(k)) break; x[k] <- x[k] / 10 }
  x
}
leer_datos <- function(ruta = "data_estudiantes.csv") {
  df <- utils::read.csv(ruta, check.names = FALSE, fileEncoding = "UTF-8-BOM", colClasses = "character", strip.white = TRUE)
  nm <- gsub("^_|_$", "", gsub("_+", "_", gsub("[^A-Za-z0-9]+", "_", trimws(names(df)))))
  nm[nm == "Curricular_units_2nd_sem_nrolled"] <- "Curricular_units_2nd_sem_enrolled"
  names(df) <- nm
  for (v in setdiff(nm, "Target"))
    df[[v]] <- if (grepl("^Curricular_units_.*_grade$", v)) limpiar_nota(df[[v]]) else as.numeric(df[[v]])
  df$Target <- factor(trimws(df$Target), levels = NIV)
  stopifnot(!anyNA(df), all(df$Curricular_units_1st_sem_grade <= 20), all(df$Curricular_units_2nd_sem_grade <= 20))
  tasa <- function(a, m) ifelse(m > 0, a / m, 0)
  df$Approval_rate_1st_sem <- tasa(df$Curricular_units_1st_sem_approved, df$Curricular_units_1st_sem_enrolled)
  df$Approval_rate_2nd_sem <- tasa(df$Curricular_units_2nd_sem_approved, df$Curricular_units_2nd_sem_enrolled)
  df
}

V_CAT <- c("Marital_status", "Application_mode", "Course", "Previous_qualification", "Nacionality",
           "Mothers_qualification", "Fathers_qualification", "Mothers_occupation", "Fathers_occupation")
V_BIN <- c("Daytime_evening_attendance", "Displaced", "Educational_special_needs", "Debtor",
           "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "International")
V_ING <- c("Application_order", "Previous_qualification_grade", "Admission_grade", "Age_at_enrollment",
           "Unemployment_rate", "Inflation_rate", "GDP")
.sem <- function(s, r) c(paste0("Curricular_units_", s, "_sem_",
                                c("credited", "enrolled", "evaluations", "approved", "grade", "without_evaluations")), r)
V_S1 <- .sem("1st", "Approval_rate_1st_sem"); V_S2 <- .sem("2nd", "Approval_rate_2nd_sem")
MOMENTOS <- c("Al ingreso" = "ingreso", "Fin del 1.er semestre" = "s1", "Fin del 2.º semestre" = "s2")
vars_momento <- function(m) list(cat = V_CAT, bin = V_BIN, num = c(V_ING, if (m %in% c("s1", "s2")) V_S1, if (m == "s2") V_S2))

CARRERAS <- c("33" = "Biocombustibles", "171" = "Animación y Multimedia", "8014" = "Servicio Social (noct.)",
  "9003" = "Agronomía", "9070" = "Diseño de Comunicación", "9085" = "Enfermería Veterinaria",
  "9119" = "Ingeniería Informática", "9130" = "Equinocultura", "9147" = "Gestión", "9238" = "Servicio Social",
  "9254" = "Turismo", "9500" = "Enfermería", "9556" = "Higiene Oral", "9670" = "Publicidad y Marketing",
  "9773" = "Periodismo", "9853" = "Educación Básica", "9991" = "Gestión (noct.)")

# ---------------------------------------------------------------------
# 2. Partición y receta (todo se ajusta SOLO con el entrenamiento)
# ---------------------------------------------------------------------
particionar <- function(df, semilla, p_test = 0.20, p_val = 0.15) {
  set.seed(semilla)
  estrat <- function(idx, p) unlist(lapply(split(idx, df$Target[idx]), function(i) i[sample.int(length(i), round(length(i) * p))]))
  te <- sort(estrat(seq_len(nrow(df)), p_test)); resto <- setdiff(seq_len(nrow(df)), te)
  va <- sort(estrat(resto, p_val)); tr <- setdiff(resto, va)
  list(train = df[tr, ], val = df[va, ], test = df[te, ], semilla = semilla)
}
ajustar_receta <- function(train, momento, umbral_raro = 0.01) {
  v <- vars_momento(momento)
  mu <- sapply(train[v$num], mean); sdv <- sapply(train[v$num], stats::sd); sdv[!is.finite(sdv) | sdv == 0] <- 1
  niveles <- lapply(v$cat, function(x) { f <- table(train[[x]]) / nrow(train); sort(as.numeric(names(f)[f >= umbral_raro])) })
  names(niveles) <- v$cat
  list(momento = momento, num = v$num, bin = v$bin, cat = v$cat, mu = mu, sd = sdv, niveles = niveles,
       prior = as.numeric(prop.table(table(train$Target))),
       tipico = c(lapply(train[unique(c(V_ING, V_S1, V_S2))], stats::median),
                  lapply(train[c(V_BIN, V_CAT)], function(z) as.numeric(names(which.max(table(z)))))))
}
codificar <- function(rec, df, v) { i <- match(df[[v]], rec$niveles[[v]]); i[is.na(i)] <- 0L; as.integer(i) } # 0 = «Otro»
x_num <- function(rec, df) {
  X <- cbind(sweep(sweep(as.matrix(df[rec$num]), 2, rec$mu), 2, rec$sd, "/"), as.matrix(df[rec$bin]))
  storage.mode(X) <- "double"; X
}
x_onehot <- function(rec, df) {
  oh <- do.call(cbind, lapply(rec$cat, function(v) {
    k <- length(rec$niveles[[v]]); m <- matrix(0, nrow(df), k + 1)
    m[cbind(seq_len(nrow(df)), codificar(rec, df, v) + 1L)] <- 1
    colnames(m) <- paste0(v, "_", c("Otro", rec$niveles[[v]])); m }))
  X <- cbind(x_num(rec, df), oh); storage.mode(X) <- "double"; X
}
x_mixto <- function(rec, df) {
  l <- lapply(rec$cat, function(v) matrix(codificar(rec, df, v), ncol = 1)); names(l) <- paste0("in_", rec$cat)
  c(l, list(in_numerico = x_num(rec, df)))
}
y_onehot <- function(y) { m <- matrix(0, length(y), 3); m[cbind(seq_along(y), as.integer(y))] <- 1; m }
pesos_clase <- function(y) { w <- length(y) / (3 * as.numeric(table(y))); w[as.integer(y)] }

# ---------------------------------------------------------------------
# 3. Modelos clásicos (se entrenan en segundos, también dentro del app)
# ---------------------------------------------------------------------
pred_base <- function(rec, n) matrix(rec$prior, n, 3, byrow = TRUE)

ajustar_logistica <- function(rec, train, pesos = TRUE) {
  X <- x_onehot(rec, train); colnames(X) <- make.names(colnames(X), unique = TRUE)
  w <- if (pesos) pesos_clase(train$Target) else rep(1, nrow(train))
  y <- stats::relevel(train$Target, ref = "Graduate")          # Graduado = referencia para los odds ratios
  nnet::multinom(y ~ ., data = data.frame(y = y, X), weights = w, decay = 1e-3, maxit = 1000, MaxNWts = 20000, trace = FALSE)
}
pred_logistica <- function(fit, rec, df) {
  X <- x_onehot(rec, df); colnames(X) <- make.names(colnames(X), unique = TRUE)
  P <- matrix(stats::predict(fit, newdata = data.frame(X), type = "probs"), ncol = 3)
  colnames(P) <- fit$lev; P[, NIV, drop = FALSE]
}
ajustar_ridge <- function(rec, train, pesos = TRUE, semilla = 1) {
  set.seed(semilla); X <- x_onehot(rec, train)
  w <- if (pesos) pesos_clase(train$Target) else rep(1, nrow(train))
  glmnet::cv.glmnet(X, train$Target, family = "multinomial", alpha = 0, weights = w, nfolds = 10, type.measure = "deviance")
}
pred_ridge <- function(fit, rec, df) matrix(stats::predict(fit, x_onehot(rec, df), s = "lambda.min", type = "response")[, NIV, 1], ncol = 3)

# Con ponderación: Random Forest balanceado (cada árbol muestrea igual número por clase; Chen, Liaw y Breiman, 2004)
ajustar_rf <- function(rec, train, pesos = TRUE, semilla = 1, ntree = 500) {
  set.seed(semilla); X <- x_onehot(rec, train); y <- train$Target
  if (pesos) { m <- min(table(y)); randomForest::randomForest(X, y, ntree = ntree, strata = y, sampsize = rep(m, 3)) }
  else randomForest::randomForest(X, y, ntree = ntree)
}
pred_rf <- function(fit, rec, df) matrix(stats::predict(fit, x_onehot(rec, df), type = "prob")[, NIV], ncol = 3)

# ---------------------------------------------------------------------
# 4. Redes neuronales: entrenamiento (keras3) y predicción en R puro
# ---------------------------------------------------------------------
dim_embedding <- function(k) as.integer(min(10, max(2, ceiling(sqrt(k + 1)))))

construir_mlp <- function(n_in, dropout = 0.3, lr = 1e-3) {
  keras3::keras_model_sequential(input_shape = n_in, name = "mlp_denso") |>
    keras3::layer_dense(128, activation = "relu", name = "d128") |> keras3::layer_dropout(dropout) |>
    keras3::layer_dense(64, activation = "relu", name = "d64") |> keras3::layer_dropout(dropout * 0.7) |>
    keras3::layer_dense(32, activation = "relu", name = "d32") |>
    keras3::layer_dense(3, activation = "softmax", name = "salida") |>
    keras3::compile(optimizer = keras3::optimizer_adam(learning_rate = lr), loss = "categorical_crossentropy", metrics = "accuracy")
}
construir_mixto <- function(rec, n_num, dropout = 0.3, lr = 1e-3) {
  ent <- list(); ramas <- list()
  for (v in rec$cat) {
    k <- length(rec$niveles[[v]]); clave <- paste0("in_", v)
    inp <- keras3::layer_input(shape = 1, dtype = "int32", name = clave)
    ramas[[clave]] <- inp |> keras3::layer_embedding(input_dim = k + 1, output_dim = dim_embedding(k), name = paste0("emb_", v)) |>
      keras3::layer_flatten()
    ent[[clave]] <- inp
  }
  inp_num <- keras3::layer_input(shape = n_num, name = "in_numerico")
  rama_num <- inp_num |> keras3::layer_dense(64, activation = "relu", name = "num64") |> keras3::layer_dropout(0.2)
  sal <- keras3::layer_concatenate(unname(c(ramas, list(rama_num)))) |>
    keras3::layer_dense(96, activation = "relu", name = "d96") |> keras3::layer_dropout(dropout) |>
    keras3::layer_dense(48, activation = "relu", name = "d48") |> keras3::layer_dropout(dropout * 0.6) |>
    keras3::layer_dense(3, activation = "softmax", name = "salida")
  m <- keras3::keras_model(inputs = c(ent, list(in_numerico = inp_num)), outputs = sal, name = "modelo_mixto")
  m |> keras3::compile(optimizer = keras3::optimizer_adam(learning_rate = lr), loss = "categorical_crossentropy", metrics = "accuracy")
  m
}
HP_RED <- list(epocas = 150, paciencia = 15, batch = 64, dropout = 0.3, lr = 1e-3)

entrenar_red <- function(tipo, rec, part, semilla, pesos = TRUE, hp = HP_RED) {
  set.seed(semilla); keras3::set_random_seed(semilla)
  try(reticulate::import("tensorflow")$config$experimental$enable_op_determinism(), silent = TRUE)
  ent <- function(df) if (tipo == "mlp") x_onehot(rec, df) else x_mixto(rec, df)
  xtr <- ent(part$train)
  modelo <- if (tipo == "mlp") construir_mlp(ncol(xtr), hp$dropout, hp$lr) else construir_mixto(rec, ncol(x_num(rec, part$train[1, ])), hp$dropout, hp$lr)
  h <- keras3::fit(modelo, xtr, y_onehot(part$train$Target), epochs = hp$epocas, batch_size = hp$batch,
                   validation_data = list(ent(part$val), y_onehot(part$val$Target)),
                   sample_weight = if (pesos) pesos_clase(part$train$Target) else NULL,
                   callbacks = list(keras3::callback_early_stopping(monitor = "val_loss", patience = hp$paciencia, restore_best_weights = TRUE)),
                   verbose = 0)
  m <- h$metrics
  capa <- function(nm) lapply(modelo$get_layer(nm)$get_weights(), function(a) { a <- as.array(a); storage.mode(a) <- "double"; a })
  pesos_r <- if (tipo == "mlp") lapply(setNames(nm = c("d128", "d64", "d32", "salida")), capa)
             else c(lapply(setNames(nm = paste0("emb_", rec$cat)), function(n) capa(n)[[1]]),
                    lapply(setNames(nm = c("num64", "d96", "d48", "salida")), capa))
  list(tipo = tipo, modelo = modelo, pesos = pesos_r, n_param = modelo$count_params(), mejor = which.min(m$val_loss),
       hist = data.frame(epoca = seq_along(m$loss), Entrenamiento = m$loss, Validación = m$val_loss))
}
.relu <- function(z) pmax(z, 0)
.densa <- function(X, w) sweep(X %*% w[[1]], 2, as.numeric(w[[2]]), "+")
.softmax <- function(Z) { Z <- Z - apply(Z, 1, max); E <- exp(Z); E / rowSums(E) }
# Paso hacia adelante en R puro (dropout inactivo en inferencia, como en Keras)
pred_red <- function(obj, rec, df) {
  p <- obj$pesos
  if (obj$tipo == "mlp") {
    h <- .relu(.densa(x_onehot(rec, df), p$d128)); h <- .relu(.densa(h, p$d64)); h <- .relu(.densa(h, p$d32))
    return(.softmax(.densa(h, p$salida)))
  }
  emb <- do.call(cbind, lapply(rec$cat, function(v) p[[paste0("emb_", v)]][codificar(rec, df, v) + 1L, , drop = FALSE]))
  num <- .relu(.densa(x_num(rec, df), p$num64))
  h <- .relu(.densa(cbind(emb, num), p$d96)); h <- .relu(.densa(h, p$d48))
  .softmax(.densa(h, p$salida))
}

# ---------------------------------------------------------------------
# 5. Evaluación
# ---------------------------------------------------------------------
auc_bin <- function(pos, s) { r <- rank(s); n1 <- sum(pos); n0 <- length(pos) - n1
  if (n1 == 0 || n0 == 0) return(NA_real_); (sum(r[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
auc_hand_till <- function(yi, P) mean(apply(utils::combn(3, 2), 2, function(p) {
  s <- yi %in% p; (auc_bin(yi[s] == p[1], P[s, p[1]]) + auc_bin(yi[s] == p[2], P[s, p[2]])) / 2 }))
metricas <- function(y, P) {
  yi <- as.integer(y); n <- length(yi); pr <- max.col(P, ties.method = "first")
  cm <- table(factor(yi, 1:3), factor(pr, 1:3))
  rec <- diag(cm) / pmax(rowSums(cm), 1); pre <- diag(cm) / pmax(colSums(cm), 1)
  f1 <- ifelse(rec + pre > 0, 2 * rec * pre / (rec + pre), 0)
  acc <- sum(diag(cm)) / n; pe <- sum(rowSums(cm) * colSums(cm)) / n^2; Pc <- pmin(pmax(P, 1e-15), 1)
  c(Exactitud = acc, `Exactitud balanceada` = mean(rec), `F1 macro` = mean(f1), Kappa = (acc - pe) / (1 - pe),
    `AUC (Hand-Till)` = auc_hand_till(yi, P), `AUC Deserción` = auc_bin(yi == 1, P[, 1]),
    `Pérdida logarítmica` = -mean(log(Pc[cbind(1:n, yi)])), Brier = mean(rowSums((P - y_onehot(y))^2)),
    `Sensib. Deserción` = rec[[1]], `Sensib. Matriculado` = rec[[2]], `Sensib. Graduado` = rec[[3]],
    `Precisión Deserción` = pre[[1]], `F1 Deserción` = f1[[1]], `F1 Matriculado` = f1[[2]], `F1 Graduado` = f1[[3]],
    `Sesgo Deserción` = mean(P[, 1]) - mean(yi == 1))   # > 0: sobreestima la deserción; < 0: la subestima
}
METRICAS_IC <- c("Exactitud balanceada", "F1 macro", "AUC (Hand-Till)", "Pérdida logarítmica")
bootstrap_ic <- function(y, P, B = 1000, semilla = 1) {
  set.seed(semilla); n <- length(y)
  M <- replicate(B, { i <- sample.int(n, n, TRUE); metricas(y[i], P[i, , drop = FALSE])[METRICAS_IC] })
  t(apply(M, 1, stats::quantile, c(0.025, 0.975), na.rm = TRUE))
}
tabla_con_ic <- function(y, lista_P, B = 1000) do.call(rbind, lapply(names(lista_P), function(nm) {
  m <- metricas(y, lista_P[[nm]]); ic <- bootstrap_ic(y, lista_P[[nm]], B)
  data.frame(Modelo = nm, Métrica = METRICAS_IC, Valor = m[METRICAS_IC], Inf95 = ic[, 1], Sup95 = ic[, 2], row.names = NULL) }))
# Diferencia pareada frente a otro modelo (p. ej. la línea base): IC bootstrap de ΔF1 y Δpérdida + McNemar exacto
comparar_pareado <- function(y, P1, P2, B = 2000, semilla = 1) {
  yi <- as.integer(y); c1 <- max.col(P1) == yi; c2 <- max.col(P2) == yi; b <- sum(c1 & !c2); cc <- sum(!c1 & c2)
  set.seed(semilla); n <- length(y); k <- c("F1 macro", "Pérdida logarítmica")
  d <- replicate(B, { i <- sample.int(n, n, TRUE); metricas(y[i], P1[i, , drop = FALSE])[k] - metricas(y[i], P2[i, , drop = FALSE])[k] })
  list(solo1 = b, solo2 = cc, p_mcnemar = if (b + cc > 0) stats::binom.test(b, b + cc)$p.value else 1,
       dif = metricas(y, P1)[k] - metricas(y, P2)[k], ic = t(apply(d, 1, stats::quantile, c(0.025, 0.975))))
}
# AUC de la clase Deserción (uno contra el resto) con IC de DeLong y prueba de DeLong pareada
roc_desercion <- function(y, P) pROC::roc(as.integer(y == "Dropout"), P[, 1], levels = c(0, 1), direction = "<", quiet = TRUE)
delong_tabla <- function(y, lista_P, ref = "Regresión logística") {
  r0 <- roc_desercion(y, lista_P[[ref]])
  do.call(rbind, lapply(names(lista_P), function(nm) { r <- roc_desercion(y, lista_P[[nm]]); ci <- as.numeric(pROC::ci.auc(r, method = "delong"))
    p <- if (nm == ref) NA else pROC::roc.test(r, r0, method = "delong", paired = TRUE)$p.value
    data.frame(Modelo = nm, `AUC Deserción` = ci[2], Inf95 = ci[1], Sup95 = ci[3], `p DeLong vs logística` = p, check.names = FALSE) }))
}
calibracion <- function(y, P, clase = 1, bins = 10) {
  p <- P[, clase]; obs <- as.integer(y) == clase; g <- cut(p, seq(0, 1, length.out = bins + 1), include.lowest = TRUE)
  d <- data.frame(conf = tapply(p, g, mean), frec = tapply(obs, g, mean), n = as.numeric(table(g))); d <- d[d$n > 0, ]
  list(tabla = d, ece = sum(d$n / length(p) * abs(d$frec - d$conf)))
}
importancia_perm <- function(predfun, df, y, vars, R = 3, semilla = 1) {
  base <- metricas(y, predfun(df))[["Pérdida logarítmica"]]; set.seed(semilla)
  sapply(vars, function(v) mean(replicate(R, { d <- df; d[[v]] <- sample(d[[v]]); metricas(y, predfun(d))[["Pérdida logarítmica"]] - base })))
}
# Odds ratios de la logística multinomial (referencia: Graduado). Numéricas: por 1 DE; binarias: 1 frente a 0.
odds_ratios <- function(fit, rec) {
  co <- stats::coef(fit); vars <- make.names(c(rec$num, rec$bin))
  do.call(rbind, lapply(rownames(co), function(cl) data.frame(Contraste = paste(NIV_ES[cl], "frente a Graduado"),
    Variable = c(rec$num, rec$bin), OR = exp(co[cl, vars]), row.names = NULL)))
}
# Prueba t corregida para validación repetida (Nadeau y Bengio, 2003)
t_corregida <- function(d, n_test, n_train) { k <- length(d); t <- mean(d) / sqrt((1 / k + n_test / n_train) * stats::var(d))
  c(t = t, p = 2 * stats::pt(-abs(t), k - 1)) }

ETIQ <- c(Marital_status = "Estado civil", Application_mode = "Modo de admisión", Application_order = "Orden de preferencia",
  Course = "Carrera", Daytime_evening_attendance = "Jornada diurna", Previous_qualification = "Formación previa",
  Previous_qualification_grade = "Nota de formación previa", Nacionality = "Nacionalidad", Mothers_qualification = "Educación de la madre",
  Fathers_qualification = "Educación del padre", Mothers_occupation = "Ocupación de la madre", Fathers_occupation = "Ocupación del padre",
  Admission_grade = "Nota de admisión", Displaced = "Desplazado", Educational_special_needs = "Necesidades especiales", Debtor = "Deudor",
  Tuition_fees_up_to_date = "Matrícula al día", Gender = "Hombre (vs. mujer)", Scholarship_holder = "Becario", Age_at_enrollment = "Edad al ingreso",
  International = "Internacional", Unemployment_rate = "Desempleo", Inflation_rate = "Inflación", GDP = "PIB",
  Curricular_units_1st_sem_credited = "1.er sem.: convalidadas", Curricular_units_1st_sem_enrolled = "1.er sem.: matriculadas",
  Curricular_units_1st_sem_evaluations = "1.er sem.: evaluaciones", Curricular_units_1st_sem_approved = "1.er sem.: aprobadas",
  Curricular_units_1st_sem_grade = "1.er sem.: nota media", Curricular_units_1st_sem_without_evaluations = "1.er sem.: sin evaluar",
  Approval_rate_1st_sem = "1.er sem.: tasa de aprobación", Curricular_units_2nd_sem_credited = "2.º sem.: convalidadas",
  Curricular_units_2nd_sem_enrolled = "2.º sem.: matriculadas", Curricular_units_2nd_sem_evaluations = "2.º sem.: evaluaciones",
  Curricular_units_2nd_sem_approved = "2.º sem.: aprobadas", Curricular_units_2nd_sem_grade = "2.º sem.: nota media",
  Curricular_units_2nd_sem_without_evaluations = "2.º sem.: sin evaluar", Approval_rate_2nd_sem = "2.º sem.: tasa de aprobación")
et <- function(v) ifelse(v %in% names(ETIQ), ETIQ[v], v)
fmt <- function(x, d = 3) formatC(x, format = "f", digits = d, decimal.mark = ",", big.mark = " ")

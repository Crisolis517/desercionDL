# =====================================================================
# Deserción estudiantil: MLP denso y modelo mixto con embeddings frente a modelos simples
# Maestría en Estadística, mención Ciencia de Datos e IA - ESPOCH
# Redes Neuronales y Deep Learning - Docente: Ms. Edwin Mejía Peñafiel
# Equipo: Shirley Adriano, Lissette Adriano, Alexis Rivera, Cristian Solís
# ---------------------------------------------------------------------
# Las redes se entrenaron en el notebook (Notebook_Proyecto2.ipynb) con keras3;
# sus pesos están en modelos_red.rds y aquí se predicen en R puro, por lo que el app
# NO necesita TensorFlow. La logística, Ridge y Random Forest se entrenan en vivo.
# Paquetes (ya instalados en shinyapps.io al desplegar; no instalar en el servidor):
# install.packages(c("shiny", "bslib", "ggplot2", "nnet", "glmnet", "randomForest", "pROC"))
# =====================================================================
suppressPackageStartupMessages({
  library(shiny); library(bslib); library(ggplot2); library(nnet); library(glmnet); library(randomForest); library(pROC)
})
source("funciones.R", encoding = "UTF-8")
datos <- leer_datos("data_estudiantes.csv")
RED <- readRDS("modelos_red.rds")
leer <- function(f) if (file.exists(f)) utils::read.csv(f, check.names = FALSE) else NULL
REF <- leer("resultados_semillas.csv"); COMP <- leer("comparaciones_pareadas.csv"); DESB <- leer("desbalance.csv")
SEMILLA <- 123
tema <- function(b = 13) theme_minimal(base_size = b) + theme(panel.grid.minor = element_blank())
tabla_fmt <- function(d, dig = 3) { for (v in names(d)) if (is.numeric(d[[v]])) d[[v]] <- fmt(d[[v]], dig); d }

tema_app <- bs_theme(version = 5, bootswatch = "flatly", primary = "#0284c7", base_font = font_google("Inter")) |>
  bs_add_rules("nav.navbar.navbar-inverse,nav.navbar.navbar-default{background-color:#0c4a6e!important}
    .navbar .navbar-brand,.navbar .nav-link{color:#e0f2fe!important}.navbar .nav-link.active{color:#fff!important;font-weight:600}
    .kpi{font-size:2rem;font-weight:800;color:#0284c7}.nota{color:#64748b;font-size:.9rem}
    .tabla-scroll{overflow-x:auto}.tabla-scroll td{white-space:nowrap;font-size:.88rem}")
caja <- function(titulo, ...) card(full_screen = TRUE, card_header(titulo), ...)
tabla <- function(id) div(class = "tabla-scroll", tableOutput(id))

panel_red <- function(id, titulo, arq) nav_panel(titulo, icon = icon(if (id == "mlp") "diagram-project" else "layer-group"),
  card(class = "p-3", h5(titulo), p(class = "nota", arq), uiOutput(paste0(id, "_info"))),
  layout_column_wrap(width = 1/2,
    caja("Curvas de pérdida del entrenamiento (línea = época restaurada)", plotOutput(paste0(id, "_curvas"), height = 300)),
    caja("Matriz de confusión en prueba (filas = clase real)", plotOutput(paste0(id, "_cm"), height = 300))),
  caja("Métricas en el conjunto de prueba", tabla(paste0(id, "_tabla"))),
  if (id == "mix") caja("Embedding aprendido de la carrera (dos primeras componentes principales)", plotOutput("plot_emb", height = 420), uiOutput("nota_emb")))

ui <- page_navbar(title = "Deserción estudiantil", theme = tema_app, id = "nav", fillable = FALSE,
  sidebar = sidebar(title = "Experimento", width = 260,
    selectInput("momento", "Momento de la alerta", MOMENTOS, selected = "s1"),
    div(class = "nota", "Partición estratificada fija (semilla 123): 68 % entrenamiento, 12 % validación, 20 % prueba. Clases ponderadas."),
    uiOutput("info_part")),

  nav_panel("Inicio", icon = icon("house"), card(class = "p-3",
    h3("¿Superan las redes neuronales a los modelos simples al predecir la deserción?"),
    p("Comparamos seis modelos para clasificar a cada estudiante en deserción, matrícula continuada o graduación: una línea base,
       la regresión logística multinomial, Ridge, Random Forest, un perceptrón multicapa (MLP) y un modelo mixto que aprende un
       embedding por variable categórica. Datos reales: Realinho et al. (2021), UCI, doi.org/10.24432/C5MC89."),
    layout_column_wrap(width = 1/4,
      card(class = "p-3 text-center", div(class = "kpi", "4 424"), "estudiantes"),
      card(class = "p-3 text-center", div(class = "kpi", "36"), "variables"),
      card(class = "p-3 text-center", div(class = "kpi", "6"), "modelos comparados"),
      card(class = "p-3 text-center", div(class = "kpi", "10"), "semillas de robustez")))),

  nav_panel("Exploración", icon = icon("chart-column"),
    layout_column_wrap(width = 1/2,
      caja("Distribución de la variable objetivo", plotOutput("plot_target", height = 280)),
      card(full_screen = TRUE, card_header(selectInput("var_explorar", NULL, setNames(c(V_ING, V_S1, V_S2, V_BIN, V_CAT), et(c(V_ING, V_S1, V_S2, V_BIN, V_CAT))),
                                                       selected = "Approval_rate_1st_sem", width = "100%")),
           plotOutput("plot_var", height = 230), uiOutput("prueba_var")))),

  nav_panel("Entrenamiento", icon = icon("gears"),
    caja("Resumen del entrenamiento de los seis modelos", tabla("tabla_entren"),
         div(class = "nota", "Las redes se entrenaron con Adam, entropía cruzada ponderada y parada temprana (paciencia 15). La logística, Ridge y Random Forest se entrenan ahora, en este servidor.")),
    layout_column_wrap(width = 1/2,
      caja("Ridge: desviación de validación cruzada según λ", plotOutput("plot_ridge", height = 280)),
      caja("Random Forest: error fuera de bolsa según el número de árboles", plotOutput("plot_rf", height = 280)))),

  panel_red("mlp", "MLP denso", "Entrada numérica + one-hot → densa 128 → densa 64 → densa 32 (ReLU, dropout) → softmax de 3 clases."),
  panel_red("mix", "Modelo mixto", "Un embedding por variable categórica + rama numérica densa 64 → concatenación → densa 96 → densa 48 → softmax de 3 clases."),

  nav_panel("Comparación", icon = icon("scale-balanced"),
    caja("Desempeño en prueba con IC 95 % bootstrap (1 000 remuestras)", tabla("tabla_comp")),
    layout_column_wrap(width = 1/2,
      caja("AUC de deserción con IC de DeLong", tabla("tabla_delong"), div(class = "nota", "p: prueba de DeLong pareada frente a la regresión logística.")),
      caja("Diferencia frente a la línea base y a la logística (ΔF1 macro, IC 95 %)", tabla("tabla_vs"))),
    layout_column_wrap(width = 1/2,
      caja("Calibración de la probabilidad de deserción", plotOutput("plot_calib", height = 320)),
      caja("Sesgo: probabilidad media predicha − proporción real de deserción", plotOutput("plot_sesgo", height = 320)))),

  nav_panel("Robustez", icon = icon("shuffle"),
    caja("10 semillas: media ± desviación estándar en prueba", tabla("tabla_ref")),
    caja("Pruebas t corregidas (Nadeau y Bengio, 2003) sobre el F1 macro", tabla("tabla_t")),
    caja("Tratamiento del desbalance: fin del 1.er semestre, 10 semillas", tabla("tabla_desb"))),

  nav_panel("Interpretación", icon = icon("magnifying-glass-chart"),
    layout_column_wrap(width = 1/2,
      caja("Odds ratios de la regresión logística (referencia: Graduado)", tabla("tabla_or"),
           div(class = "nota", "Numéricas: por 1 desviación estándar. Se omiten las variables académicas muy colineales (aprobadas, tasa, nota).")),
      caja("Importancia de Gini (Random Forest)", plotOutput("plot_gini", height = 380))),
    card(full_screen = TRUE, card_header(div(class = "d-flex gap-3 align-items-center", "Importancia por permutación (aumento de la pérdida logarítmica)",
      selectInput("mod_imp", NULL, MODELOS[-1], selected = "Modelo mixto", width = "220px"))), plotOutput("plot_imp", height = 400))),

  nav_panel("Predicción", icon = icon("user-graduate"),
    layout_sidebar(sidebar = sidebar(width = 320,
        selectInput("p_carrera", "Carrera", setNames(names(CARRERAS), CARRERAS), selected = "9500"),
        layout_column_wrap(width = 1/2, selectInput("p_genero", "Género", c("Mujer" = 0, "Hombre" = 1)), numericInput("p_edad", "Edad", 20, 16, 70)),
        numericInput("p_admision", "Nota de admisión (95–190)", 127, 95, 190),
        layout_column_wrap(width = 1/2, selectInput("p_beca", "Becario", c("No" = 0, "Sí" = 1)), selectInput("p_deudor", "Deudor", c("No" = 0, "Sí" = 1))),
        selectInput("p_pago", "Matrícula al día", c("Sí" = 1, "No" = 0)),
        h6("1.er semestre"), layout_column_wrap(width = 1/3, numericInput("p_m1", "Matric.", 6, 0, 26), numericInput("p_a1", "Aprob.", 5, 0, 26), numericInput("p_n1", "Nota", 12, 0, 20)),
        h6("2.º semestre"), layout_column_wrap(width = 1/3, numericInput("p_m2", "Matric.", 6, 0, 23), numericInput("p_a2", "Aprob.", 5, 0, 20), numericInput("p_n2", "Nota", 12, 0, 20)),
        actionButton("btn_pred", "Predecir con los seis modelos", class = "btn-primary w-100", icon = icon("bolt"))),
      uiOutput("nota_pred"), caja("Probabilidad por clase según cada modelo", plotOutput("plot_pred", height = 340)), card(tabla("tabla_pred")))),

  nav_panel("Metodología", icon = icon("book"), card(class = "p-3",
    h4("Diseño"), p("Variable respuesta: estado final (deserción, matriculado, graduado). Tres momentos de alerta con la información disponible en cada uno.
      Partición estratificada 68/12/20; todo el preprocesamiento se ajusta solo con el entrenamiento; categorías con menos del 1 % agrupadas en «Otro»."),
    h4("Modelos"), p("Línea base (prevalencia); regresión logística multinomial (decay 0,001); Ridge multinomial (glmnet, λ por validación cruzada de 10 pliegues);
      Random Forest de 500 árboles balanceado por clase; MLP 128-64-32; modelo mixto con embeddings de dimensión min(10, ⌈√(k+1)⌉).
      Desbalance: ponderación por la inversa de la frecuencia de clase."),
    h4("Evaluación"), p("F1 macro (selección del modelo), exactitud balanceada, AUC de Hand y Till, AUC de deserción con IC de DeLong, pérdida logarítmica,
      sensibilidad por clase, sesgo y calibración. IC 95 % bootstrap; McNemar; DeLong pareado; 10 semillas con prueba t corregida de Nadeau y Bengio (2003)."),
    h4("Referencias"), tags$ul(
      tags$li("Realinho, V., Vieira Martins, M., Machado, J. y Baptista, L. (2021). Predict Students' Dropout and Academic Success. UCI. doi.org/10.24432/C5MC89"),
      tags$li("Guo, C. y Berkhahn, F. (2016). Entity embeddings of categorical variables. arXiv:1604.06737."),
      tags$li("Chen, C., Liaw, A. y Breiman, L. (2004). Using random forest to learn imbalanced data. University of California, Berkeley."),
      tags$li("DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. Biometrics, 44(3), 837–845."),
      tags$li("Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52, 239–281."),
      tags$li("Hand, D. J. y Till, R. J. (2001). A simple generalisation of the area under the ROC curve for multiple class classification problems. Machine Learning, 45, 171–186.")))))

server <- function(input, output, session) {
  ex <- reactive({ part <- particionar(datos, SEMILLA); list(part = part, rec = RED[[input$momento]]$rec, red = RED[[input$momento]]) })
  output$info_part <- renderUI({ p <- ex()$part; div(class = "nota mt-2", sprintf("Entrenamiento %d · validación %d · prueba %d", nrow(p$train), nrow(p$val), nrow(p$test))) })

  clasicos <- reactive({ e <- ex(); tr <- e$part$train
    withProgress(message = "Entrenando logística, Ridge y Random Forest…", value = .2, {
      t <- system.time({ lg <- ajustar_logistica(e$rec, tr); incProgress(.2); rg <- ajustar_ridge(e$rec, tr, TRUE, SEMILLA); incProgress(.3)
                         rf <- ajustar_rf(e$rec, tr, TRUE, SEMILLA) })[["elapsed"]]
      list(lg = lg, rg = rg, rf = rf, seg = t) }) })
  predictores <- reactive({ e <- ex(); c <- clasicos()
    list(`Línea base` = function(d) pred_base(e$rec, nrow(d)), `Regresión logística` = function(d) pred_logistica(c$lg, e$rec, d),
         Ridge = function(d) pred_ridge(c$rg, e$rec, d), `Random Forest` = function(d) pred_rf(c$rf, e$rec, d),
         `MLP denso` = function(d) pred_red(e$red$mlp, e$rec, d), `Modelo mixto` = function(d) pred_red(e$red$mixto, e$rec, d)) })
  probs <- reactive({ te <- ex()$part$test; lapply(predictores(), function(f) f(te)) })
  yte <- reactive(ex()$part$test$Target)

  # --- Exploración ---
  output$plot_target <- renderPlot(res = 96, {
    d <- as.data.frame(table(Clase = factor(NIV_ES[as.character(datos$Target)], NIV_ES)))
    ggplot(d, aes(Clase, Freq, fill = Clase)) + geom_col(width = .6) + geom_text(aes(label = sprintf("%d (%s %%)", Freq, fmt(100 * Freq / sum(Freq), 1))), vjust = -.4) +
      scale_fill_manual(values = unname(COL_CLASE)) + scale_y_continuous(expand = expansion(c(0, .12))) + tema() + theme(legend.position = "none") + labs(x = NULL, y = "Estudiantes")
  })
  output$plot_var <- renderPlot(res = 96, {
    v <- input$var_explorar; d <- data.frame(x = datos[[v]], Clase = factor(NIV_ES[as.character(datos$Target)], NIV_ES))
    if (v %in% c(V_CAT, V_BIN)) {
      if (v == "Course") d$x <- CARRERAS[as.character(d$x)]
      top <- names(sort(table(d$x), decreasing = TRUE))[1:min(8, length(unique(d$x)))]; d$x <- ifelse(as.character(d$x) %in% top, as.character(d$x), "Otras")
      ggplot(d, aes(x = x, fill = Clase)) + geom_bar(position = "fill") + coord_flip() + scale_fill_manual(values = unname(COL_CLASE)) +
        scale_y_continuous(labels = function(z) paste0(100 * z, " %")) + tema(12) + labs(x = NULL, y = "Proporción", fill = NULL)
    } else ggplot(d, aes(Clase, x, fill = Clase)) + geom_boxplot(alpha = .85, outlier.alpha = .3) + scale_fill_manual(values = unname(COL_CLASE)) +
      tema(12) + theme(legend.position = "none") + labs(x = NULL, y = et(v))
  })
  output$prueba_var <- renderUI({ v <- input$var_explorar; x <- datos[[v]]; y <- datos$Target
    txt <- if (v %in% c(V_CAT, V_BIN)) { t <- table(x, y); cs <- suppressWarnings(chisq.test(t))
      sprintf("χ² = %s; p %s; V de Cramér = %s", fmt(cs$statistic, 1), if (cs$p.value < .001) "< 0,001" else paste("=", fmt(cs$p.value)), fmt(sqrt(cs$statistic / (sum(t) * (min(dim(t)) - 1))))) }
    else { k <- kruskal.test(x ~ y); sprintf("Kruskal-Wallis H = %s; p %s; ε² = %s", fmt(k$statistic, 1), if (k$p.value < .001) "< 0,001" else paste("=", fmt(k$p.value)), fmt((k$statistic - 2) / (length(x) - 3))) }
    div(class = "nota px-3 pb-2", txt) })

  # --- Entrenamiento ---
  output$tabla_entren <- renderTable({ e <- ex(); c <- clasicos()
    data.frame(Modelo = MODELOS, Dónde = c("Sin entrenamiento", rep("En vivo (este servidor)", 3), rep("Notebook (keras3)", 2)),
      Detalle = c("Prevalencia del entrenamiento", "Multinomial, decay 0,001", sprintf("λ = %s (validación cruzada)", fmt(c$rg$lambda.min, 4)),
                  sprintf("500 árboles; error OOB = %s", fmt(tail(c$rf$err.rate[, 1], 1))),
                  sprintf("%s parámetros; época %d", fmt(e$red$mlp$n_param, 0), e$red$mlp$mejor), sprintf("%s parámetros; época %d", fmt(e$red$mixto$n_param, 0), e$red$mixto$mejor))) })
  output$plot_ridge <- renderPlot(res = 96, { c <- clasicos()$rg; d <- data.frame(l = log(c$lambda), m = c$cvm, lo = c$cvlo, hi = c$cvup)
    ggplot(d, aes(l, m)) + geom_ribbon(aes(ymin = lo, ymax = hi), fill = "#99f6e4", alpha = .5) + geom_line(colour = "#0f766e") +
      geom_vline(xintercept = log(c$lambda.min), linetype = 2) + tema(12) + labs(x = "log(λ)", y = "Desviación multinomial (VC)") })
  output$plot_rf <- renderPlot(res = 96, { e <- clasicos()$rf$err.rate; d <- data.frame(n = seq_len(nrow(e)), OOB = e[, "OOB"])
    ggplot(d, aes(n, OOB)) + geom_line(colour = "#b45309") + tema(12) + labs(x = "Árboles", y = "Error fuera de bolsa") })

  # --- Redes ---
  for (id in c("mlp", "mix")) local({ id <- id; key <- if (id == "mlp") "mlp" else "mixto"; nm <- if (id == "mlp") "MLP denso" else "Modelo mixto"
    output[[paste0(id, "_info")]] <- renderUI({ o <- ex()$red[[key]]
      div(class = "alert alert-light", sprintf("%s parámetros. Entrenada en el notebook: mejores pesos en la época %d de %d (parada temprana).", fmt(o$n_param, 0), o$mejor, nrow(o$hist))) })
    output[[paste0(id, "_curvas")]] <- renderPlot(res = 96, { o <- ex()$red[[key]]
      h <- rbind(data.frame(e = o$hist$epoca, C = "Entrenamiento", L = o$hist$Entrenamiento), data.frame(e = o$hist$epoca, C = "Validación", L = o$hist$Validación))
      ggplot(h, aes(e, L, colour = C)) + geom_line(linewidth = 1) + geom_vline(xintercept = o$mejor, linetype = 2) +
        scale_colour_manual(values = c("#94a3b8", COL_MODELO[[nm]])) + tema(12) + labs(x = "Época", y = "Entropía cruzada", colour = NULL) })
    output[[paste0(id, "_cm")]] <- renderPlot(res = 96, { P <- probs()[[nm]]; y <- yte()
      cm <- as.data.frame(table(Real = factor(NIV_ES[as.character(y)], NIV_ES), Predicho = factor(NIV_ES[max.col(P)], NIV_ES)))
      cm$prop <- ave(cm$Freq, cm$Real, FUN = function(z) z / sum(z))
      ggplot(cm, aes(Predicho, Real, fill = prop)) + geom_tile(colour = "white") +
        geom_text(aes(label = sprintf("%d\n(%s %%)", Freq, fmt(100 * prop, 0))), colour = ifelse(cm$prop > .5, "white", "#0f172a")) +
        scale_fill_gradient(low = "#e0f2fe", high = "#1d4ed8", limits = c(0, 1), guide = "none") + scale_y_discrete(limits = rev) + tema(12) })
    output[[paste0(id, "_tabla")]] <- renderTable({ m <- metricas(yte(), probs()[[nm]]); data.frame(Métrica = names(m), Valor = fmt(m)) })
  })
  output$plot_emb <- renderPlot(res = 96, { e <- ex(); W <- e$red$mixto$pesos$emb_Course; tr <- e$part$train
    niv <- c("Otro", e$rec$niveles$Course); cod <- codificar(e$rec, tr, "Course"); n <- tabulate(cod + 1, nrow(W))
    tasa <- as.numeric(tapply(tr$Target == "Dropout", factor(cod, 0:(nrow(W) - 1)), mean)); pc <- prcomp(W)$x
    d <- data.frame(PC1 = pc[, 1], PC2 = pc[, 2], lab = ifelse(niv == "Otro", "Otro", CARRERAS[as.character(niv)]), tasa = tasa, n = n); d <- d[d$n > 0, ]
    ggplot(d, aes(PC1, PC2, colour = tasa, size = n)) + geom_point() + geom_text(aes(label = lab), size = 3.4, vjust = -1.1, colour = "#334155") +
      scale_colour_gradient(low = "#0284c7", high = "#dc2626", labels = function(z) paste0(round(100 * z), " %"), name = "Deserción") +
      scale_size(range = c(2, 7), name = "Estudiantes") + tema(12) + labs(x = "Componente 1", y = "Componente 2") })
  output$nota_emb <- renderUI({ e <- ex(); W <- e$red$mixto$pesos$emb_Course; tr <- e$part$train; cod <- codificar(e$rec, tr, "Course")
    n <- tabulate(cod + 1, nrow(W)); tasa <- tapply(tr$Target == "Dropout", factor(cod, 0:(nrow(W) - 1)), mean); ok <- n >= 10
    rho <- abs(cor(prcomp(W)$x[ok, 1], tasa[ok], method = "spearman"))
    div(class = "nota px-3 pb-2", sprintf("|ρ de Spearman| entre la componente 1 y la tasa de deserción de cada carrera = %s: la red ordenó las carreras según su riesgo sin que se le indicara.", fmt(rho, 2))) })

  # --- Comparación ---
  output$tabla_comp <- renderTable({ t <- withProgress(message = "Bootstrap…", tabla_con_ic(yte(), probs(), 1000))
    t$E <- sprintf("%s [%s; %s]", fmt(t$Valor), fmt(t$Inf95), fmt(t$Sup95))
    w <- reshape(t[c("Modelo", "Métrica", "E")], idvar = "Modelo", timevar = "Métrica", direction = "wide"); names(w) <- sub("^E\\.", "", names(w)); w })
  output$tabla_delong <- renderTable(tabla_fmt(delong_tabla(yte(), probs()[-1])))
  output$tabla_vs <- renderTable({ l <- probs(); y <- yte()
    do.call(rbind, lapply(MODELOS[-(1:2)], function(nm) do.call(rbind, lapply(c("Línea base", "Regresión logística"), function(ref) {
      r <- comparar_pareado(y, l[[nm]], l[[ref]], B = 1000)
      data.frame(Modelo = nm, `Frente a` = ref, `ΔF1 [IC 95 %]` = sprintf("%s [%s; %s]", fmt(r$dif[[1]]), fmt(r$ic[1, 1]), fmt(r$ic[1, 2])), check.names = FALSE) })))) })
  output$plot_calib <- renderPlot(res = 96, { l <- probs()[-1]; y <- yte()
    d <- do.call(rbind, lapply(names(l), function(nm) cbind(calibracion(y, l[[nm]])$tabla, Modelo = nm)))
    ggplot(d, aes(conf, frec, colour = Modelo)) + geom_abline(linetype = 2, colour = "grey60") + geom_line() + geom_point() + coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
      scale_colour_manual(values = COL_MODELO) + tema(12) + labs(x = "Probabilidad predicha", y = "Frecuencia observada", colour = NULL) })
  output$plot_sesgo <- renderPlot(res = 96, { l <- probs()[-1]; y <- yte()
    d <- data.frame(Modelo = factor(names(l), rev(MODELOS)), s = sapply(l, function(P) metricas(y, P)[["Sesgo Deserción"]]))
    ggplot(d, aes(s, Modelo, fill = Modelo)) + geom_col() + geom_vline(xintercept = 0) + scale_fill_manual(values = COL_MODELO, guide = "none") +
      scale_x_continuous(labels = function(z) fmt(z, 3)) + tema(12) + labs(x = "> 0 sobreestima · < 0 subestima", y = NULL) })

  # --- Robustez ---
  output$tabla_ref <- renderTable({ validate(need(!is.null(REF), "Falta resultados_semillas.csv")); REF[REF$Momento == names(MOMENTOS)[MOMENTOS == input$momento], ] })
  output$tabla_t <- renderTable({ validate(need(!is.null(COMP), "Falta comparaciones_pareadas.csv"))
    d <- COMP[COMP$Métrica == "F1 macro" & COMP$Momento == names(MOMENTOS)[MOMENTOS == input$momento], c("Comparación", "Diferencia media", "Gana el 1.º", "p")]; tabla_fmt(d) })
  output$tabla_desb <- renderTable({ validate(need(!is.null(DESB), "Falta desbalance.csv")); tabla_fmt(DESB) })

  # --- Interpretación ---
  output$tabla_or <- renderTable({ o <- odds_ratios(clasicos()$lg, ex()$rec)
    o <- o[o$Variable %in% c("Tuition_fees_up_to_date", "Debtor", "Scholarship_holder", "Gender", "Age_at_enrollment", "Admission_grade", "Unemployment_rate"), ]
    o$Variable <- et(o$Variable); tabla_fmt(o) })
  output$plot_gini <- renderPlot(res = 96, { g <- sort(importance(clasicos()$rf)[, 1], decreasing = TRUE)[1:12]
    v <- sub("_(Otro|[0-9]+)$", "", names(g)); lab <- ifelse(v %in% names(ETIQ), paste0(et(v), ifelse(grepl("_[0-9]+$|_Otro$", names(g)), paste0(" = ", sub(".*_", "", names(g))), "")), names(g))
    ggplot(data.frame(v = lab, g = g), aes(g, reorder(v, g))) + geom_col(fill = "#b45309") + tema(12) + labs(x = "Disminución media de Gini", y = NULL) })
  output$plot_imp <- renderPlot(res = 96, { e <- ex(); f <- predictores()[[input$mod_imp]]; te <- e$part$test
    imp <- withProgress(message = "Permutando variables…", importancia_perm(f, te, te$Target, c(e$rec$num, e$rec$bin, e$rec$cat), R = 3))
    d <- head(data.frame(v = et(names(imp)), i = imp)[order(-imp), ], 15)
    ggplot(d, aes(i, reorder(v, i))) + geom_col(fill = COL_MODELO[[input$mod_imp]]) + scale_x_continuous(labels = function(z) fmt(z, 2)) + tema(12) +
      labs(x = "Aumento de la pérdida logarítmica en prueba", y = NULL) })

  # --- Predicción ---
  pred <- eventReactive(input$btn_pred, { e <- ex(); x <- as.data.frame(e$rec$tipico)
    x$Course <- as.numeric(input$p_carrera); x$Gender <- as.numeric(input$p_genero); x$Age_at_enrollment <- input$p_edad; x$Admission_grade <- input$p_admision
    x$Scholarship_holder <- as.numeric(input$p_beca); x$Debtor <- as.numeric(input$p_deudor); x$Tuition_fees_up_to_date <- as.numeric(input$p_pago)
    a1 <- min(input$p_a1, input$p_m1); a2 <- min(input$p_a2, input$p_m2)
    x$Curricular_units_1st_sem_enrolled <- input$p_m1; x$Curricular_units_1st_sem_approved <- a1; x$Curricular_units_1st_sem_grade <- input$p_n1
    x$Approval_rate_1st_sem <- if (input$p_m1 > 0) a1 / input$p_m1 else 0
    x$Curricular_units_2nd_sem_enrolled <- input$p_m2; x$Curricular_units_2nd_sem_approved <- a2; x$Curricular_units_2nd_sem_grade <- input$p_n2
    x$Approval_rate_2nd_sem <- if (input$p_m2 > 0) a2 / input$p_m2 else 0
    l <- lapply(predictores(), function(f) f(x))
    do.call(rbind, lapply(names(l), function(nm) data.frame(Modelo = nm, Clase = NIV_ES, Prob = as.numeric(l[[nm]])))) })
  output$nota_pred <- renderUI(div(class = "alert alert-light", HTML(paste0("Modelos del momento «", names(MOMENTOS)[MOMENTOS == input$momento], "». Las variables que no están en el formulario toman la mediana o la moda del entrenamiento.",
    if (input$momento == "ingreso") " Los datos de los semestres <b>no</b> se usan." else if (input$momento == "s1") " Los datos del 2.º semestre <b>no</b> se usan." else ""))))
  output$plot_pred <- renderPlot(res = 96, { d <- pred(); d$Clase <- factor(d$Clase, NIV_ES); d$Modelo <- factor(d$Modelo, MODELOS)
    ggplot(d, aes(Clase, Prob, fill = Modelo)) + geom_col(position = position_dodge(.85), width = .8) + scale_fill_manual(values = COL_MODELO) +
      scale_y_continuous(labels = function(z) paste0(100 * z, " %"), limits = c(0, 1)) + tema(12) + labs(x = NULL, y = "Probabilidad", fill = NULL) + theme(legend.position = "bottom") })
  output$tabla_pred <- renderTable({ d <- pred(); d$Prob <- paste0(fmt(100 * d$Prob, 1), " %")
    w <- reshape(d, idvar = "Modelo", timevar = "Clase", direction = "wide"); names(w) <- c("Modelo", NIV_ES); w })
}
shinyApp(ui, server)

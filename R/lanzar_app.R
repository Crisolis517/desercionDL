#' Abre el dashboard del proyecto
#'
#' @param ... Argumentos que se pasan a shiny::runApp (por ejemplo, port).
#' @return No devuelve valor; abre la aplicación Shiny.
#' @examples
#' \dontrun{desercionDL::lanzar_app()}
lanzar_app <- function(...) {
  dir <- system.file("app", package = "desercionDL")
  if (dir == "") stop("No se encontró la carpeta de la aplicación. Reinstale el paquete.")
  shiny::runApp(dir, ...)
}

// La niebla del fondo reacciona al puntero y a lo que pasa en el formulario.
// Aquí solo se escriben variables CSS: quien anima es el navegador, y siempre
// sobre opacity / translate / scale, que no obligan a repintar los gradientes.
(() => {
  const raiz = document.documentElement;
  const quietud = matchMedia("(prefers-reduced-motion: reduce)");

  // Posición a la que tiende la niebla, y la que tiene ahora mismo. La segunda
  // persigue a la primera con retraso: el humo llega tarde, no salta.
  let destinoX = 0.5, destinoY = 0.5;
  let x = 0.5, y = 0.5;
  let corriendo = false;

  const paso = () => {
    x += (destinoX - x) * 0.04;
    y += (destinoY - y) * 0.04;
    raiz.style.setProperty("--niebla-x", x.toFixed(4));
    raiz.style.setProperty("--niebla-y", y.toFixed(4));

    // Se para sola al alcanzar al puntero; el siguiente movimiento la arranca
    // otra vez. Así no queda un rAF girando en vacío.
    if (Math.abs(destinoX - x) > 0.0005 || Math.abs(destinoY - y) > 0.0005) {
      requestAnimationFrame(paso);
    } else {
      corriendo = false;
    }
  };

  const arrancar = () => {
    if (corriendo || quietud.matches) return;
    corriendo = true;
    requestAnimationFrame(paso);
  };

  addEventListener("pointermove", (e) => {
    destinoX = e.clientX / innerWidth;
    destinoY = e.clientY / innerHeight;
    arrancar();
  }, { passive: true });

  // Si el puntero se va de la ventana, la niebla vuelve despacio al centro.
  document.addEventListener("pointerleave", () => {
    destinoX = destinoY = 0.5;
    arrancar();
  });

  const form = document.getElementById("apertura");
  if (!form) return;

  // Se espesa mientras se escribe: el fondo acompaña, no distrae.
  form.addEventListener("focusin", () => raiz.classList.add("niebla-densa"));
  form.addEventListener("focusout", () => raiz.classList.remove("niebla-densa"));

  // Y se revuelve al repartir las cartas. La vuelta la hace la transición CSS,
  // de ahí que baste con quitar la clase.
  form.addEventListener("submit", () => {
    raiz.classList.add("niebla-revuelta");
    setTimeout(() => raiz.classList.remove("niebla-revuelta"), 2400);
  });
})();

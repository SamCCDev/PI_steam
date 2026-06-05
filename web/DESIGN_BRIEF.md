# Brief de diseño — SteamPredict v2 (para iterar el frontend en un artifact de Claude)

Pega este documento en claude.ai y pídele que **rediseñe/pula la interfaz**. Importante: el artifact corre
**aislado, sin red**, así que **no** puede llamar al backend real. Trabaja con **datos mock** (incluidos abajo)
y entrega un **único HTML auto-contenido**. Luego ese diseño se re-cablea a los endpoints reales en `web/`.

## Contexto

Dashboard de un predictor de éxito comercial de videojuegos en Steam. El usuario arma un juego hipotético
(precio, géneros, etiquetas, idiomas, experiencia del estudio…) y ve la predicción de tres modelos de ML.
Es para una exposición universitaria. Debe verse profesional, denso pero legible, **estética Steam**.

## Estética

- Paleta Steam: fondo `#171a21`, paneles `#1b2838` / `#16202d`, bordes `#2a475e`, acento `#66c0f4`,
  texto `#c7d5e0`, texto tenue `#8f98a0`.
- Colores de clase: Flop `#c75450` (rojo), Rentable `#66c0f4` (azul), Hit `#a4d007` (verde lima).
- Tipografía system-ui. Dark mode. Esquinas redondeadas suaves, bordes sutiles, sombras discretas.

## Restricciones técnicas (respetar)

- Gráficos con **ECharts** (`https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js`).
- Si usa Tailwind, que sea **v4** y los **colores custom se definen en CSS (`@theme` o CSS plano), NO por
  `tailwind.config` JS** (en v4 eso se ignora).
- **Sin `fetch`**: usar las constantes mock de abajo. Un solo archivo HTML que renderice solo.
- Mantener las **5 vistas** y una **consola/terminal inferior** de actividad (monoespaciada, log con colores).

## Las 5 vistas

1. **Simulador** — panel de inputs a la izquierda (sliders, selects, chips de género/tag/categoría/plataforma)
   y resultados a la derecha: KPIs (clasificación, prob. de no-Flop, owners estimados, ingreso bruto), barras
   de probabilidad de las 3 clases y un bloque de incertidumbre.
2. **Comparar modelos** — barras agrupadas (3 modelos × 3 clases) + tabla con AUC/F1/accuracy.
3. **Juegos del mismo camino** — tarjetas de juegos reales parecidos, con su clase y owners.
4. **Recomendaciones** — KPIs antes/después de P(Hit) + lista del "mejor paquete" de cambios + cambios
   individuales con su delta.
5. **Panel analítico** — grilla de gráficos: dona de mercado, histograma de precios, owners por género,
   importancia de variables, scatter precio/owners, matriz de confusión, % Hit por trimestre.

Dos modos: **Presentación** (todo) y **Quick** (formulario compacto). Buscador de juegos reales arriba.

## Datos mock (formas reales del backend)

```js
const MOCK_CONFIG = {
  classes: ["Flop","Rentable","Hit"],
  thresholds: { flop_max: 200000, hit_min: 1000000 },
  models: ["lr","svm","mlp"],
  numeric: {  // 12 numéricas; ejemplo de una:
    price: { median: 5.79, min: 0, max: 200, label: "Price" },
    supported_languages: { median: 3, min: 1, max: 30, label: "Supported Languages" }
    // + total_achievements, total_dlcs, min_ram_gb, short_desc_len, dev_game_count,
    //   pub_game_count, num_genres, num_tags, num_categories, dev_success_prior
  },
  categorical: {
    dev_experience: { categories: ["AAA","Establecido","Novato"], default: "Novato" },
    controller_support: { categories: ["full","none","partial"], default: "none" },
    pub_experience: { categories: ["AAA","Establecido","Novato"], default: "Novato" },
    release_quarter: { categories: ["Q1","Q2","Q3","Q4","Desconocido"], default: "Q4" },
    price_tier: { categories: ["Budget","F2P","Mid","Premium"], default: "Budget" }
  },
  groups: {  // chips; cada uno { col, label, prevalencia, mutable }
    genre: [{col:"genre_action",label:"Action",prevalencia:0.51,mutable:true}, /* 13 */],
    cat:   [{col:"cat_single_player",label:"Single Player",prevalencia:0.91,mutable:false}, /* 13 */],
    tag:   [{col:"tag_rpg",label:"RPG",prevalencia:0.29,mutable:true}, /* 44 */],
    platform: [{col:"platform_windows",label:"Windows",prevalencia:0.99,mutable:false}, /* 3 */]
  }
};

const MOCK_PREDICT = {
  modelos: {
    lr:  { probs: { Flop:0.117, Rentable:0.536, Hit:0.347 }, clase:"Rentable" },
    svm: { probs: { Flop:0.189, Rentable:0.679, Hit:0.132 }, clase:"Rentable" },
    mlp: { probs: { Flop:0.049, Rentable:0.851, Hit:0.100 }, clase:"Rentable" }
  },
  owners_estimados: 270734,
  clase_por_owners: "Rentable",
  incertidumbre: { margen_top2: 0.315, desacuerdo: false, nivel: "media", modelo_ref: "mlp" }
};

const MOCK_RECOMMEND = {
  modelo: "mlp", phit_antes: 0.100, phit_despues: 0.429, delta: 0.329,
  cambios: [
    { feature:"tag_multiplayer", desc:"Añadir 'Multiplayer'", phit:0.207 },
    { feature:"tag_sandbox",     desc:"Añadir 'Sandbox'",     phit:0.308 },
    { feature:"tag_difficult",   desc:"Añadir 'Difficult'",   phit:0.429 }
  ],
  cambios_individuales: [
    { desc:"Añadir 'Multiplayer'", delta:0.107 },
    { desc:"Lanzar en Q4 (temporada alta)", delta:0.041 },
    { desc:"Ajustar precio a $19.99", delta:0.018 }
  ]
};

const MOCK_SIMILAR = { juegos: [
  { appid:235900, name:"RPG Maker XP", developer:"Enterbrain", clase:"Rentable", owners:200000, price:24.99, similitud:0.82 },
  { appid:6910,   name:"Deus Ex: GOTY", developer:"Eidos",     clase:"Hit",      owners:2000000, price:6.99, similitud:0.78 },
  { appid:22380,  name:"Fallout: New Vegas", developer:"Obsidian", clase:"Hit",  owners:5000000, price:9.99, similitud:0.74 }
]};

const MOCK_STATS = {
  market: [{clase:"Flop",n:2144},{clase:"Rentable",n:4234},{clase:"Hit",n:1439}],
  price_hist: { labels:["F2P/<5","5–10","10–15","15–20","20–30","30–40","40–60","60+"],
                counts:[2600,1400,900,800,1100,500,300,200] },
  owners_by_genre: [ {genero:"Massively Multiplayer",n:473,owners_mediana:500000,pct_hit:31.3},
                     {genero:"RPG",n:1990,owners_mediana:300000,pct_hit:24.1} /* ~12 */ ],
  lr_importance: { positivos:[{feature:"num tags",coef:0.62},{feature:"Multiplayer",coef:0.41}],
                   negativos:[{feature:"price",coef:-0.33},{feature:"Casual",coef:-0.28}] },
  metrics: { lr:{auc_ovr_macro:0.884,f1_macro:0.692,accuracy:0.697},
             svm:{auc_ovr_macro:0.881,f1_macro:0.720,accuracy:0.739},
             mlp:{auc_ovr_macro:0.888,f1_macro:0.740,accuracy:0.772},
             _meta:{features:92} },
  confusion: { mlp:{ labels:["Flop","Rentable","Hit"], matrix:[[296,121,12],[65,737,45],[21,93,174]] } },
  scatter: [ [19.99, 250000, "Rentable"], [4.99, 1500000, "Hit"], [29.99, 80000, "Flop"] /* ~500 */ ],
  hit_by_quarter: [ {q:"Q1",pct_hit:20.8,pct_norentable:53.3},{q:"Q2",pct_hit:20.8,pct_norentable:52.3},
                    {q:"Q3",pct_hit:20.7,pct_norentable:55.9},{q:"Q4",pct_hit:22.9,pct_norentable:57.3} ],
  n_total: 7817
};
```

## Qué pedirle que mejore

- Jerarquía visual y espaciado más cuidados; transiciones suaves entre vistas y al actualizar resultados.
- Que los gráficos se vean tipo Power BI (limpios, con buen contraste sobre el fondo oscuro).
- Micro-interacciones en los chips y KPIs.
- Que el resultado sea **un solo `.html`** (estilos y JS incluidos) para pegarlo de vuelta y re-cablear.

## Cómo vuelve

Cuando tengas el diseño que te guste, me pasas el HTML resultante y yo reemplazo el `fetch` mock por las
llamadas reales a `/api/*` y lo integro en `web/`, conservando la lógica que ya funciona.

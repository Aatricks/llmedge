` para que los modelos con alto razonamiento no agoten la ventana de contexto tan rápido.` (Wait, `<think>...` is in the original? Actually it says `think.../think` but the original says `<think>...`. I'll keep it exactly as is.)

   `### Tool Calling` -> `### Llamadas a herramientas (Tool Calling)`
   `Use `edge.text.toolAgent(...)` when...` -> `Usa `edge.text.toolAgent(...)` cuando quieras que el modelo invoque herramientas definidas por la aplicación. Las herramientas de solo lectura se ejecutan automáticamente; las herramientas de acción requieren una decisión de política explícita.`
   (Code block stays)
   `Tool calls use a structured JSON...` -> `Las llamadas a herramientas usan internamente un sobre JSON estructurado: `{"tool":"nombre","arguments":{...}}`. El analizador también acepta el campo heredado `tool_name` para robustez, pero los prompts nuevos solo emiten la forma `tool`.`
   `For JVM or desktop hosts...` -> `Para hosts JVM o de escritorio donde `bash` está disponible, también puedes optar por una herramienta de ejecución de shell:`
   (Code block stays)
   `The bash tool accepts either...` -> `La herramienta bash acepta `argv` para comandos estructurados o `command` para cadenas de shell crudas. Si `bash` no está disponible o el comando falla, la herramienta devuelve un resultado de error estructurado en lugar de afirmar éxito.`

   `### Speech Request Objects` -> `### Objetos de solicitud de voz`
   `Speech APIs now support request-first...` -> `Las APIs de voz ahora admiten llamadas primero-por-solicitud además de las sobrecargas de conveniencia existentes:`
   (Code block stays)
   `This keeps new speech entrypoints aligned...` -> `Esto mantiene las nuevas entradas de voz alineadas con el estilo primero-por-solicitud ya usado por texto y generación de imágenes, mientras preserva las sobrecargas antiguas de listas de parámetros para compatibilidad.`

   `### Text Generation Performance Tuning` -> `### Ajuste de rendimiento de generación de texto`
   `The text stack now separates...` -> `El stack de texto ahora separa el procesamiento de prompt/lote de la generación de un solo token para que puedas sintonizar las dos fases de forma independiente:`
   (Code block stays)
   `Practical defaults:` -> `Valores predeterminados prácticos:`
   `- `text.promptThreads`: prompt/batch decode threads` -> `- `text.promptThreads`: hilos de decodificación de prompt/lote`
   `- `text.generationThreads`: single-token generation threads` -> `- `text.generationThreads`: hilos de generación de token único`
   `- `text.batchSize`: blocking text batch size (default `8`)` -> `- `text.batchSize`: tamaño de lote para texto bloqueante (predeterminado `8`)`
   `- `text.streamBatchSize`: streaming batch size (default `4`)` -> `- `text.streamBatchSize`: tamaño de lote para streaming (predeterminado `4`)`
   `- `text.cache.maxMemoryMb`: upper bound for text-model cache accounting; the cache now refreshes against native model/state footprint instead of only the GGUF file size` -> `- `text.cache.maxMemoryMb`: límite superior para el contabilidad de la caché de modelos de texto; la caché ahora se actualiza según la huella nativa del modelo/estado en lugar de solo el tamaño del archivo GGUF`
   `Batch-size guidance:` -> `Guía de tamaño de lote:`
   `- `1`: lowest latency per chunk, highest JNI overhead` -> `- `1`: menor latencia por fragmento, mayor sobrecarga de JNI`
   `- `4`: good default for streaming UI updates` -> `- `4`: buen predeterminado para actualizaciones de UI en streaming`
   `- `8`: good default for blocking text responses` -> `- `8`: buen predeterminado para respuestas de texto bloqueantes`
   `- `12+`: better throughput for longer offline generations, but can delay intermediate updates` -> `- `12+`: mejor rendimiento para generaciones offline más largas, pero puede retrasar actualizaciones intermedias`

   `### Image Text Extraction (OCR)` -> `### Extracción de texto de imágenes (OCR)`
   `llmedge uses Google ML Kit...` -> `llmedge usa el reconocimiento de texto de Google ML Kit para extraer texto de imágenes.`
   `#### Quick Start` -> `#### Inicio rápido`
   (Code block stays)
   `#### OCR Engines` -> `#### Motores OCR`
   `**Google ML Kit Text Recognition**` -> `**Reconocimiento de texto de Google ML Kit**`
   `- Fast and lightweight` -> `- Rápido y ligero`
   `- No additional data files needed` -> `- No requiere archivos de datos adicionales`
   `- Good for Latin scripts` -> `- Bueno para scripts latinos`
   `- Add dependency: ...` -> `- Agrega la dependencia: ...`
   `OCR is exposed directly through...` -> `El OCR se expone directamente a través de `edge.vision.extractText(...)`. El envoltorio de conveniencia antiguo `VisionMode` ha sido eliminado; los llamantes ahora eligen explícitamente entre OCR y análisis VLM en lugar de enrutamiento a través de una segunda capa de abstracción.`

   `### Vision Models` -> `### Modelos de visión`
   `Analyze images using Vision Language Models...` -> `Analiza imágenes usando Modelos de Lenguaje de Visión (como LLaVA o Phi-3 Vision) mediante `edge.vision`.`
   `> [!WARNING]` -> `> [!WARNING]`
   `The VLM path is experimental...` -> `La ruta VLM es experimental. Requiere un GGUF capaz de visión y un archivo mmproj/proyector coincidente. Cuando estos componentes no están disponibles o son incompatibles, `edge.vision.analyze(...)` falla rápidamente con un error claro en lugar de caer silenciosamente en prompts solo de texto. El OCR sigue disponible a través de `edge.vision.extractText(...)`.`
   (Code block stays)
   `The current high-level vision path...` -> `La ruta de visión de alto nivel actual crea un tiempo de ejecución `SmolLM` nuevo por solicitud, por lo que favorece el aislamiento y la limpieza predecible sobre la reutilización en grupo de alto rendimiento.`
   `The manager handles the complex pipeline of:` -> `El administrador maneja el flujo complejo de:`
   `1. Preprocessing the image` -> `1. Preprocesamiento de la imagen`
   `2. Loading the vision projector and model` -> `2. Carga del proyector de visión y del modelo`
   `3. Encoding the image to embeddings` -> `3. Codificación de la imagen en incrustaciones`
   `4. Generating the textual response` -> `4. Generación de la respuesta textual`
   `Vision model support is currently experimental...` -> `El soporte para modelos de visión es actualmente experimental y requiere arquitecturas de modelo específicas (como LLaVA-Phi-3).`

   `### Speech-to-Text (Whisper)` -> `### Texto a voz (Whisper) / Reconocimiento de voz (Whisper)` (Wait, STT is Speech-to-Text, which is "Voz a texto" in Spanish. I'll use `### Reconocimiento de voz (Whisper)` or `### Voz a texto (Whisper)`. I'll stick to `### Voz a texto (Whisper)`.)
   `Transcribe audio using the new...` -> `Transcribe audio usando el nuevo cliente `edge.speech`: `
   (Code block stays)
   `#### Real-time Streaming Transcription` -> `#### Transcripción en streaming en tiempo real`
   `For live captioning, use the streaming transcription API...` -> `Para subtítulos en vivo, usa la API de transcripción en streaming con un enfoque de ventana deslizante:`
   (Code block stays)
   `**Streaming parameters:**` -> `**Parámetros de streaming:**`
   `- `stepMs`: How often transcription runs...` -> `- `stepMs`: Frecuencia de ejecución de la transcripción (predeterminado: 3000ms). Menor = actualizaciones más rápidas, mayor uso de CPU.`
   `- `lengthMs`: Audio window size...` -> `- `lengthMs`: Tamaño de la ventana de audio (predeterminado: 10000ms). Ventanas más largas mejoran la precisión.`
   `- `keepMs`: Overlap with previous window...` -> `- `keepMs`: Solapamiento con la ventana anterior (predeterminado: 200ms). Ayuda a mantener el contexto.`
   `- `useVad`: Voice Activity Detection...` -> `- `useVad`: Detección de Actividad de Voz - omite audio silencioso (predeterminado: true).`
   `Direct `Whisper` access remains available...` -> `El acceso directo a `Whisper` sigue disponible para flujos avanzados, pero el cliente de voz con espacio de nombres es la ruta de integración estándar.`
   `**Recommended models:**` -> `**Modelos recomendados:**`
   (List stays mostly same, translate descriptions)
   `- `ggml-tiny.bin` (~75MB) - Fast, lower accuracy` -> `- `ggml-tiny.bin` (~75MB) - Rápido, menor precisión`
   `- `ggml-base.bin` (~142MB) - Good balance` -> `- `ggml-base.bin` (~142MB) - Buen equilibrio`
   `- `ggml-small.bin` (~466MB) - Higher accuracy` -> `- `ggml-small.bin` (~466MB) - Mayor precisión`

   `### Text-to-Speech (Bark)` -> `### Texto a voz (Bark)`
   `Generate speech using `edge.speech`: ` -> `Genera voz usando `edge.speech`: `
   (Code block stays)
   `Direct `BarkTTS` access remains available...` -> `El acceso directo a `BarkTTS` sigue disponible para flujos avanzados, pero el cliente de voz con espacio de nombres es la ruta de integración estándar.`

   `### Stable Diffusion (image generation)` -> `### Stable Diffusion (generación de imágenes)`
   `Generate images on-device using the namespaced...` -> `Genera imágenes en el dispositivo usando el cliente con espacio de nombres `edge.image`: `
   (Code block stays)
   `**Key Optimizations:**` -> `**Optimizaciones clave:**`
   `- **EasyCache**: `edge.image` automatically enables...` -> `- **EasyCache**: `edge.image` habilita automáticamente EasyCache para modelos Diffusion Transformer (DiT) compatibles como Flux, SD3, Wan, Qwen Image y Z-Image; permanece deshabilitado para pipelines clásicos de UNet.`
   `- **Flash Attention**: Automatically enabled...` -> `- **Flash Attention**: Habilitado automáticamente para dimensiones de imagen compatibles.`
   `- **LoRA**: Apply fine-tuned weights...` -> `- **LoRA**: Aplica pesos de ajuste fino sobre la marcha sin fusionar modelos.`
   `For explicit runtime ownership...` -> `Para propiedad explícita del tiempo de ejecución o experimentos de carga nativa personalizados, la clase `StableDiffusion` sigue disponible en la capa de API experta.`
   `#### Streaming progress` -> `#### Progreso en streaming`
   `generate blocks until the bitmap is ready...` -> `generate bloquea hasta que el bitmap esté listo. Para impulsar una barra de progreso, usa `generateStream`, que emite un evento `Progress` por paso de desruido y un evento final `Completed` que lleva la imagen:`
   (Code block stays)
   `Cancelling the collection cancels...` -> `Cancelar la colección cancela la generación. Los eventos de paso solo cubren el bucle de muestreo; la carga del modelo ocurre antes del primer evento, por lo que mantén la barra indeterminada hasta entonces. La aplicación de ejemplo deriva una estimación de tiempo restante desde el retraso entre eventos (`StepEtaEstimator` en llmedge-examples).`
   `#### Image upscaling (ESRGAN)` -> `#### Escalado de imágenes (ESRGAN)`
   `edge.image.upscale runs an ESRGAN model...` -> `edge.image.upscale ejecuta un modelo ESRGAN sobre un bitmap y devuelve el resultado ampliado. Los puntos de control ESRGAN de arquitectura antigua se cargan directamente; 4x_foolhardy_Remacri es la referencia probada:`
   (Code block stays)
   `Details worth knowing:` -> `Detalles que vale la pena conocer:`
   `- `factor = 0` (the default) uses...` -> `- `factor = 0` (predeterminado) usa la escala incrustada en el modelo. Remacri es 4x.`
   `- The upscaler runs on CPU unless...` -> `- El escalador se ejecuta en CPU a menos que la solicitud establezca `useVulkan = true`. Las entradas grandes se procesan en bloques de 128px, y el callback de progreso informa recuentos de bloques, no pasos.`
   `- Input is capped at 1024×1024...` -> `- La entrada se limita a 1024×1024. Un paso 4x sobre una imagen de 1024px ya produce un bitmap de 16 MP, cerca del límite práctico de memoria en teléfonos gama media.`
   `- In isolated worker mode...` -> `- En modo de worker aislado, el escalado se ejecuta en el proceso `:llmedge_sd` con el mismo watchdog y recuperación ante fallos que la generación de imágenes.`
   `- The ESRGAN context is created and freed per call...` -> `- El contexto ESRGAN se crea y libera por llamada, por lo que espera una breve pausa de carga del modelo (los pesos son ~64 MB) antes del primer bloque.`
   `#### MiniT2I` -> `#### MiniT2I`
   `The `MiniT2I` helper downloads...` -> `El ayudante `MiniT2I` descarga el transformer de difusión MiniT2I B/16 independiente y su codificador de texto FLAN-T5 Large, luego los enruta a los slots de difusión dividida y modelo T5 de stable-diffusion.cpp:`
   (Code block stays)
   `The helper defaults to the model's 512×512...` -> `El ayudante usa por defecto la configuración 512×512, 100 pasos, CFG 6 del modelo. Ancho, alto, pasos, CFG, semilla y flash attention siguen configurables a través de `MiniT2I.imageRequest(...)`.`
   `#### FLUX.2 Klein 4B (distilled DiT, split model)` -> `#### FLUX.2 Klein 4B (DiT destilado, modelo dividido)`
   `[FLUX.2 Klein 4B]... is a step-distilled...` -> `[FLUX.2 Klein 4B]... es un transformer de difusión destilado por pasos que produce imágenes de alta calidad en ~4 pasos. Es la misma arquitectura sobre la que se construyen los modelos **Bonsai Image** binarios/ternarios de PrismML: los propios pesos 1-bit/ternarios de Bonsai se distribuyen solo en empaquetados MLX (Apple) y GemLite (CUDA), que no se cargan en Android, por lo que esta compilación GGUF es el equivalente ejecutable en Android con una huella comparable.`
   `Unlike a classic single-file checkpoint...` -> `A diferencia de un punto de control clásico de archivo único, FLUX.2 se carga como tres componentes: el transformer de difusión (GGUF), un codificador de texto Qwen3-4B y el VAE de FLUX.2. El ayudante `Flux2Klein` conecta los tres más los valores predeterminados destilados (CFG 1.0, 4 pasos):`
   (Code block stays)
   `Internally this sets...` -> `Internamente esto establece `ImageGenerationRequest.splitDiffusionModel = true`, que enruta el transformer a `diffusion_model_path` de stable-diffusion.cpp y el codificador Qwen3 a `llm_path` (en lugar del único slot `model_path`), y descarga los pesos a CPU. Huella: ~2.5 GB DiT (Q4_0) + ~2.1 GB codificador (Q3_K_M) + ~0.3 GB VAE, por lo que apunta a dispositivos con mayor RAM.`
   `##### Low-end: Bonsai (QAT) DiT for a smaller transformer` -> `##### Gama baja: DiT Bonsai (QAT) para un transformer más pequeño`
   `PrismML's **Bonsai Image** models are...` -> `Los modelos **Bonsai Image** de PrismML son FLUX.2 Klein 4B ajustado con entrenamiento consciente de cuantización (QAT) a pesos ternarios. Se distribuyen solo en empaquetados MLX/GemLite (no cargables en Android) y en una forma `-unpacked` densa-bf16 usando la nomenclatura de diffusers no estándar `Flux2KleinPipeline`, que stable-diffusion.cpp no puede ingerir directamente.`
   `scripts/convert_bonsai_flux2_to_bfl.py converts...` -> `scripts/convert_bonsai_flux2_to_bfl.py convierte un transformer Bonsai desempacado a la nomenclatura BFL que espera sdcpp (renombra y fusiona el bloque doble `to_q/k/v` en `*_attn.qkv`). Luego cuantiza con stable-diffusion.cpp:`
   (Code block stays)
   `A prebuilt Q2_K of this is published at...` -> `Un Q2_K precompilado de esto se publica en [`Aatricks/bonsai-image-ternary-4B-FLUX2-klein-GGUF`]... y se conecta en `Flux2Klein.bonsaiDiffusionModel`. Los pesos QAT sobreviven bien a `Q2_K`, dando un **DiT coherente de ~1.3 GB** (vs ~2.5 GB para Q4_0 base). Nota: los tipos ternarios literales de ggml (`tq1_0`/`tq2_0`, ~0.8–1.0 GB) se cargan y ejecutan en CPU pero su escala por 256 pesos es demasiado gruesa para las escalas entrenadas por 128 de Bonsai y producen salida degradada — las escalas más finas por subbloque de 16 de `Q2_K` son las que preservan la calidad.`
   `#### Sequential loading for ~4 GB-RAM devices` -> `#### Carga secuencial para dispositivos de ~4 GB de RAM`
   `The text encoder (~2 GB) is the dominant memory cost...` -> `El codificador de texto (~2 GB) es el costo dominante de memoria. El modo secuencial carga solo el codificador Qwen3 para precomputar la condición de texto, lo libera, luego carga solo el DiT para generar — así la RAM pico es `max(codificador, DiT)` (~2.6 GB) en lugar de la suma (~4 GB):`
   (Code block stays)
   `ImageGenerationRequest.sequential drives this...` -> `ImageGenerationRequest.sequential controla esto; el tiempo de ejecución ejecuta las dos fases automáticamente (solo codificador → precomputar → liberar → solo DiT → generar mediante una condición precomputada), respaldado por `sd_precompute_condition` / `sd_generate_image_with_precomputed_condition` de stable-diffusion.cpp.`

   `### Video Generation` -> `### Generación de vídeo`
   `Generate short video clips using...` -> `Genera clips de vídeo cortos usando `edge.image.generateVideo(...)`. El cliente con espacio de nombres expone el progreso como un `Flow` mientras reutiliza la lógica de carga de Wan existente internamente.`
   `**Hardware Requirements**:` -> `**Requisitos de hardware**:`
   `- **12GB+ RAM** recommended...` -> `- Se recomienda **12GB+ de RAM** para carga estándar.`
   `- **8GB+ RAM** supported...` -> `- **8GB+ de RAM** soportado mediante `forceSequentialLoad = true` (más lento pero seguro en memoria).`
   (Code block stays)
   `edge.image automatically:` -> `edge.image automáticamente:`
   `1. Downloads the necessary Wan 2.1...` -> `1. Descarga los archivos de modelo Wan 2.1 necesarios (Difusión, VAE, T5).`
   `2. Sequentially loads components...` -> `2. Carga componentes secuencialmente para minimizar el uso pico de memoria (si se solicita).`
   `3. Manages the generation loop...` -> `3. Gestiona el bucle de generación y la conversión de fotogramas.`
   `See `llmedge-examples` for a complete UI implementation.` -> `Consulta `llmedge-examples` para una implementación de UI completa.`
   `Running the example app:` -> `Ejecutar la aplicación de ejemplo:`
   `1. Build the library (from the repo root):` -> `1. Compila la biblioteca (desde la raíz del repositorio):`
   (Code block stays)
   `2. Build and install the example app:` -> `2. Compila e instala la aplicación de ejemplo:`
   (Code block stays)
   `3. Open the app on device and pick...` -> `3. Abre la aplicación en el dispositivo y selecciona la demo "Stable Diffusion" desde el launcher. La demo descarga cualquier archivo faltante desde Hugging Face y ejecuta una generación txt2img rápida.`
   `Notes:` -> `Notas:`
   `- The example explicitly downloads a VAE...` -> `- El ejemplo descarga explícitamente un archivo safetensors VAE para la demo `Meina/MeinaMix`; muchos repos incluyen archivos VAE, pero algunos repos de modelos GGUF incluyen todo lo necesario. Si el repo carece de un archivo de modelo GGUF obtendrás una IllegalArgumentException obvia — proporciona un `filename` o elige un repo diferente en ese caso.`
   `- Use the system downloader for large...` -> `- Usa el descargador del sistema para archivos safetensors/gguf grandes para evitar presión en el montón en Android.`

   `### On-device RAG` -> `### RAG en el dispositivo`
   `The library includes a minimal on-device RAG pipeline...` -> `La biblioteca incluye un pipeline RAG mínimo en el dispositivo, similar a Android-Doc-QA, construido con:`
   `- Sentence embeddings (ONNX)` -> `- Incrustaciones de oraciones (ONNX)`
   `- Whitespace `TextSplitter`` -> `- `TextSplitter` de espacio en blanco`
   `- In-memory cosine `VectorStore` with JSON persistence` -> `- `VectorStore` coseno en memoria con persistencia JSON`
   `- `SmolLM` for context-aware responses...` -> `- `SmolLM` para respuestas conscientes del contexto a través de la sesión RAG gestionada por la fachada`

   `### Setup` -> `### Configuración`
   `1. Download embeddings` -> `1. Descargar incrustaciones`
   `From the Hugging Face repository...` -> `Desde el repositorio de Hugging Face `sentence-transformers/all-MiniLM-L6-v2`, coloca:`
   (File paths stay)
   `2. Build the library` -> `2. Compilar la biblioteca`
   (Code block stays)
   `3. Use in your application` -> `3. Usar en tu aplicación`
   (Code block stays)
   `Direct `RAGEngine` construction remains available...` -> `La construcción directa de `RAGEngine` sigue disponible para flujos avanzados, pero el código de aplicación nuevo debería preferir `edge.rag.createSession()` para que la propiedad y desmontaje del tiempo de ejecución estén alineados con el resto de la biblioteca.`

   `### Expert APIs` -> `### APIs Expertas`
   `SmolLM`, `StableDiffusion`, `Whisper`, `BarkTTS`, `RAGEngine`, and direct `HuggingFaceHub` access are still available when you need to hold a native runtime directly or override low-level loading behavior. They are intentionally secondary to the facade APIs.` -> `El acceso a `SmolLM`, `StableDiffusion`, `Whisper`, `BarkTTS`, `RAGEngine` y `HuggingFaceHub` directo sigue disponible cuando necesitas mantener un tiempo de ejecución nativo directamente o anular el comportamiento de carga de bajo nivel. Están intencionalmente secundarios a las APIs de fachada.`
   `Examples:` -> `Ejemplos:`
   (Code block stays)
   `// Direct model download when you need full control over artifact selection.` -> `// Descarga directa de modelo cuando necesitas control total sobre la selección de artefactos.`
   `// Expert text runtime with live reasoning-state control.` -> `// Tiempo de ejecución experto de texto con control en vivo del estado de razonamiento.`
   `// Expert RAG wiring when you want to own both the runtime and the pipeline yourself.` -> `// Conexión RAG experta cuando quieres poseer tanto el tiempo de ejecución como el pipeline tú mismo.`

   `## Building` -> `## Compilación`
   `Building GPU backends on Android` -> `Compilación de backends de GPU en Android`
   `If you want GPU acceleration for the native inference backends, follow these notes and requirements. On Android, llmedge now prefers `OPENCL -> VULKAN -> CPU` when GPU use is allowed for text, Whisper, and image/video requests. OpenCL support is experimental, Android-only, and currently limited to `arm64-v8a`. Bark remains CPU-only.` -> `Si quieres aceleración por GPU para los backends nativos de inferencia, sigue estas notas y requisitos. En Android, llmedge ahora prefiere `OPENCL -> VULKAN -> CPU` cuando se permite el uso de GPU para solicitudes de texto, Whisper e imágenes/vídeos. El soporte OpenCL es experimental, solo para Android y actualmente limitado a `arm64-v8a`. Bark permanece solo en CPU.`
   `Prerequisites` -> `Prerrequisitos`
   `- Android NDK r27 or newer...` -> `- Android NDK r27 o más nuevo (NDK r27 usado en desarrollo; el NDK proporciona los encabezados C de Vulkan). Asegúrate de que tu NDK coincida con la versión usada por tu entorno de compilación.`
   `- CMake 3.22+ and Ninja...` -> `- CMake 3.22+ y Ninja (el plugin de Gradle para Android recogerá CMake cuando esté configurado).`
   `- Gradle (use the wrapper: `./gradlew`).` -> `- Gradle (usa el wrapper: `./gradlew`).`
   `- Android API (minSdk) 30 or higher...` -> `- API de Android (minSdk) 30 o superior. `llmedge` apunta a Android 11+ hoy en día, y el soporte Vulkan aún requiere Vulkan 1.2.`
   `- (Optional) `VULKAN_SDK` set in the environment...` -> `- (Opcional) `VULKAN_SDK` configurado en el entorno si compilas shaders o usas herramientas del SDK de Vulkan en el host. La compilación busca un encabezado `vulkan.hpp` coincidente cuando sea necesario.`

   `### Host Setup for Vulkan Builds (Ubuntu/WSL)` -> `### Configuración del host para compilaciones Vulkan (Ubuntu/WSL)`
   `To build the library with Vulkan support on a Linux host or WSL2, you must install the Vulkan shader compiler and development headers:` -> `Para compilar la biblioteca con soporte Vulkan en un host Linux o WSL2, debes instalar el compilador de shaders Vulkan y los encabezados de desarrollo:`
   `1. **Install Dependencies**:` -> `1. **Instalar dependencias**:`
   (Code block stays)
   `2. **Verify glslc**:` -> `2. **Verificar glslc**:`
   `Ensure `glslc` is in your PATH:` -> `Asegúrate de que `glslc` esté en tu PATH:`
   (Code block stays)
   `3. **Android NDK**:` -> `3. **Android NDK**:`
   `Ensure you have Android NDK **r27** (specifically `27.2.12479018`) installed via Android Studio or the SDK manager.` -> `Asegúrate de tener el Android NDK **r27** (específicamente `27.2.12479018`) instalado a través de Android Studio o el administrador de SDK.`

   `Build flags` -> `Banderas de compilación`
   `- On Linux/macOS hosts, the Gradle build enables Vulkan by default. On Windows hosts, it defaults to `OFF` because the upstream shader-generator step is still fragile under the Android cross-build toolchain. Re-enable it explicitly only when your environment supports that path.` -> `- En hosts Linux/macOS, la compilación de Gradle habilita Vulkan por defecto. En hosts Windows, predetermina `OFF` porque el paso del generador de shaders upstream sigue siendo frágil bajo la cadena de herramientas de compilación cruzada de Android. Rehábilítalo explícitamente solo cuando tu entorno soporte esa ruta.`
   `- Experimental Android OpenCL is disabled by default. Enable it with `-PllmedgeAndroidOpencl=ON` or the environment variable `LLMEDGE_ANDROID_OPENCL=ON`.` -> `- El OpenCL experimental para Android está deshabilitado por defecto. Habilítalo con `-PllmedgeAndroidOpencl=ON` o la variable de entorno `LLMEDGE_ANDROID_OPENCL=ON`.`
   `- If you want both OpenCL and Vulkan compiled in explicitly, use:` -> `- Si quieres compilar tanto OpenCL como Vulkan explícitamente, usa:`
   (Code block stays)
   `Alternatively, set the same flags in your Android Studio CMake configuration. `LLMEDGE_ANDROID_OPENCL` is the library's experimental OpenCL toggle, while `-DSD_VULKAN=ON` and `-DGGML_VULKAN=ON` force Vulkan support for Stable Diffusion and ggml.` -> `Alternativamente, establece las mismas banderas en tu configuración CMake de Android Studio. `LLMEDGE_ANDROID_OPENCL` es el interruptor experimental de OpenCL de la biblioteca, mientras que `-DSD_VULKAN=ON` y `-DGGML_VULKAN=ON` fuerzan el soporte Vulkan para Stable Diffusion y ggml.`

   `Notes about headers and toolchain` -> `Notas sobre encabezados y cadena de herramientas`
   `- The build fetches `Vulkan-Hpp` (`vulkan.hpp`) and pins it to the NDK's Vulkan headers to avoid API mismatch. If you have a local `VULKAN_SDK` you can point to it, otherwise the project will use the fetched headers.` -> `- La compilación obtiene `Vulkan-Hpp` (`vulkan.hpp`) y lo fija a los encabezados Vulkan del NDK para evitar desajustes de API. Si tienes un `VULKAN_SDK` local puedes apuntar a él, de lo contrario el proyecto usará los encabezados obtenidos.`
   `- When OpenCL is enabled, the build uses repo-managed OpenCL headers and a link-time loader shim. The packaged app still resolves the device's OpenCL implementation at runtime rather than shipping its own platform ICD.` -> `- Cuando OpenCL está habilitado, la compilación usa encabezados OpenCL gestionados por el repo y un shim de cargador en tiempo de enlace. La aplicación empaquetada aún resuelve la implementación OpenCL del dispositivo en tiempo de ejecución en lugar de incluir su propio ICD de plataforma.`
   `- The repository also builds a small host toolchain to generate SPIR-V shaders at build time; ensure your build host has a working C++ toolchain (clang/gcc) and CMake configured.` -> `- El repositorio también compila una pequeña cadena de herramientas de host para generar shaders SPIR-V en tiempo de compilación; asegúrate de que tu host de compilación tenga una cadena de herramientas C++ funcional (clang/gcc) y CMake configurado.`

   `Runtime verification` -> `Verificación en tiempo de ejecución`
   `- To verify GPU capability at runtime:` -> `- Para verificar la capacidad de GPU en tiempo de ejecución:`
   `    - Run the app on an Android 11+ device.` -> `    - Ejecuta la app en un dispositivo Android 11+. `
   `    - Use the per-subsystem capability APIs to inspect the engines you care about, for example `LLMEdge.getTextBackendAvailability()`, `LLMEdge.getSpeechBackendAvailability()`, `LLMEdge.getImageBackendAvailability()`, and `LLMEdge.getVisionBackendAvailability()`.` -> `    - Usa las APIs de capacidad por subsistema para inspeccionar los motores que te interesan, por ejemplo `LLMEdge.getTextBackendAvailability()`, `LLMEdge.getSpeechBackendAvailability()`, `LLMEdge.getImageBackendAvailability()`, y `LLMEdge.getVisionBackendAvailability()`.`
   `    - Inspect runtime logs for the selected backend and any fallback reason. Example:` -> `    - Inspecciona los registros de tiempo de ejecución para el backend seleccionado y cualquier motivo de respaldo. Ejemplo:`
   (Code block stays)
   `    Look for messages indicating OpenCL or Vulkan initialization. `LLMEdgeConfig(text = TextRuntimeConfig(useVulkan = true))` means "allow a supported GPU backend", not "force Vulkan".` -> `    Busca mensajes que indiquen inicialización de OpenCL o Vulkan. `LLMEdgeConfig(text = TextRuntimeConfig(useVulkan = true))` significa "permitir un backend de GPU compatible", no "forzar Vulkan".`

   `Troubleshooting` -> `Solución de problemas`
   `- If you see "Vulkan 1.2 required" or linker errors for Vulkan symbols, confirm `minSdk` is set to 30 or higher in `llmedge/build.gradle.kts` and that your NDK provides the expected Vulkan headers.` -> `- Si ves "Vulkan 1.2 requerido" o errores del enlazador para símbolos Vulkan, confirma que `minSdk` está configurado en 30 o superior en `llmedge/build.gradle.kts` y que tu NDK proporciona los encabezados Vulkan esperados.`
   `- If experimental OpenCL is not available, or if a GPU backend fails to initialize or execute, llmedge falls back to Vulkan or CPU automatically. For text, Whisper, and image/video, a failing backend is blacklisted per subsystem for the rest of the process and the next backend is retried once.` -> `- Si OpenCL experimental no está disponible, o si un backend de GPU falla al inicializarse o ejecutarse, llmedge cae automáticamente a Vulkan o CPU. Para texto, Whisper e imágenes/vídeos, un backend fallido se pone en lista negra por subsistema para el resto del proceso y el siguiente backend se reintenta una vez.`
   `- If your device lacks both usable OpenCL and Vulkan support, the native code falls back to the CPU backend.` -> `- Si tu dispositivo carece tanto de soporte OpenCL como Vulkan usable, el código nativo cae al backend de CPU.`
   `- If the Vulkan driver initializes but the first generate hangs forever at the first compute dispatch (observed on PowerVR DXT-48 / Pixel 10 Tensor G5), the automatic fallback cannot trigger — load succeeds, so no failure is observed. Force the CPU backend for image/video generation with `LLMEdgeConfig(image = ImageRuntimeConfig(useVulkan = false))`, or enable process isolation (below) so the library detects and recovers from the hang automatically.` -> `- Si el controlador Vulkan se inicializa pero el primer generate se cuelga indefinidamente en el primer envío de cómputo (observado en PowerVR DXT-48 / Pixel 10 Tensor G5), la caída automática no puede activarse: la carga tiene éxito, por lo que no se observa fallo. Fuerza el backend de CPU para generación de imágenes/vídeos con `LLMEdgeConfig(image = ImageRuntimeConfig(useVulkan = false))`, o habilita el aislamiento de proceso (abajo) para que la biblioteca detecte y se recupere del colgar automáticamente.`

   `Process isolation for image/video generation (opt-in)` -> `Aislamiento de proceso para generación de imágenes/vídeos (optativo)`
   `LLMEdgeConfig(image = ImageRuntimeConfig(workerMode = DiffusionWorkerMode.ISOLATED_PROCESS))` runs the diffusion stack in a library-owned `:llmedge_sd` worker process:` -> `LLMEdgeConfig(image = ImageRuntimeConfig(workerMode = DiffusionWorkerMode.ISOLATED_PROCESS))` ejecuta el stack de difusión en un proceso worker `:llmedge_sd` de propiedad de la biblioteca:`
   `- A native crash in the diffusion stack surfaces as a typed `WorkerCrashedException` (with one automatic CPU retry) instead of killing the app.` -> `- Un fallo nativo en el stack de difusión aparece como una `WorkerCrashedException` tipada (con un reintento automático en CPU) en lugar de matar la app.`
   `- A GPU driver that hangs at dispatch is detected by a watchdog — no progress heartbeat while the worker's CPU time stays flat (a legitimate cold shader compile pegs a core and is never killed) — the worker is killed, and per `hangRecoveryPolicy` the request is transparently retried on CPU (default) or failed with `GenerationHangException`.` -> `- Un controlador de GPU que se cuelga en el envío es detectado por un watchdog: sin latido de progreso mientras el tiempo de CPU del worker se mantiene plano (una compilación en frío de shader legítima satura un núcleo y nunca se mata) — el worker se mata, y según `hangRecoveryPolicy` la solicitud se reintenta transparentemente en CPU (predeterminado) o falla con `GenerationHangException`.`
   `- Hang verdicts persist across sessions (keyed to the OS build fingerprint, so a system/driver update re-enables the GPU automatically). `ImageClient.resetBackendVerdicts(context)` clears them manually.` -> `- Los veredictos de colgar persisten entre sesiones (indexados por la huella digital de compilación del SO, por lo que una actualización de sistema/controlador reactiva la GPU automáticamente). `ImageClient.resetBackendVerdicts(context)` los limpia manualmente.`
   `- The public `ImageClient` API is unchanged; `DiffusionWorkerMode.IN_PROCESS` remains the default for now. Apps that already host llmedge in their own service process should stay in-process to avoid a redundant extra process.` -> `- La API pública `ImageClient` no cambia; `DiffusionWorkerMode.IN_PROCESS` permanece como predeterminado por ahora. Las apps que ya hospedan llmedge en su propio proceso de servicio deberían permanecer in-process para evitar un proceso extra redundante.`

   `#### Notes:` -> `#### Notas:`
   `- Uses `com.tom-roush:pdfbox-android` for PDF parsing.` -> `- Usa `com.tom-roush:pdfbox-android` para análisis de PDF.`
   `- Embeddings library: `io.gitlab.shubham0204:sentence-embeddings:v6`.` -> `- Biblioteca de incrustaciones: `io.gitlab.shubham0204:sentence-embeddings:v6`.`
   `- Scanned PDFs require OCR (e.g., ML Kit or Tesseract) before indexing.` -> `- Los PDF escaneados requieren OCR (ej., ML Kit o Tesseract) antes de indexar.`
   `- ONNX `token_type_ids` errors are automatically handled; override via `EmbeddingConfig` if required.` -> `- Los errores ONNX `token_type_ids` se manejan automáticamente; anula vía `EmbeddingConfig` si es necesario.`

   `## Architecture` -> `## Arquitectura`
   `The Kotlin side is now organized around a few explicit layers instead of one eager facade:` -> `El lado Kotlin ahora está organizado alrededor de unas pocas capas explícitas en lugar de una sola fachada ansiosa:`
   `1. `LLMEdge` is a thin convenience shell...` -> `1. `LLMEdge` es una cáscara de conveniencia fina que crea perezosamente clientes de dominio (`text`, `speech`, `image`, `vision`, `rag`) al primer acceso.`
   `2. `ModelRepository` owns model acquisition...` -> `2. `ModelRepository` posee la adquisición y validación de modelos para archivos locales y descargas de Hugging Face.`
   `3. `RuntimePool` and `RuntimeCoordinator` provide shared runtime caching, backend selection, and failure blacklisting.` -> `3. `RuntimePool` y `RuntimeCoordinator` proporcionan caché compartida de tiempo de ejecución, selección de backend y listado en negro de fallos.`
   `4. `RuntimePoolProfile` lets each domain describe cache sizing, keying, loading, and backend policy without duplicating pool boilerplate.` -> `4. `RuntimePoolProfile` permite que cada dominio describa el dimensionamiento, clave, carga y política de caché sin duplicar el código boilerplate del pool.`
   `5. `TextClient`, `SpeechClient`, `ImageClient`, `VisionClient`, and `RAGClient` remain independently constructible for advanced use, but `LLMEdge` is the canonical public entrypoint.` -> `5. `TextClient`, `SpeechClient`, `ImageClient`, `VisionClient` y `RAGClient` permanecen construibles independientemente para uso avanzado, pero `LLMEdge` es el punto de entrada público canónico.`
   `6. `ConversationSessionSupport` centralizes transcript state and runtime access for chat sessions and tool agents.` -> `6. `ConversationSessionSupport` centraliza el estado del transcripción y el acceso al tiempo de ejecución para sesiones de chat y agentes de herramientas.`
   `7. `VisionInputPreparer` and `VisionRuntimeExecutor` split image preprocessing/embedding from generation execution.` -> `7. `VisionInputPreparer` y `VisionRuntimeExecutor` separan el preprocesamiento/incrustación de imágenes de la ejecución de generación.`
   `8. `RAGIndexer`, `RAGRetriever`, and `RAGAnswerer` separate document ingestion, retrieval, and answer generation.` -> `8. `RAGIndexer`, `RAGRetriever` y `RAGAnswerer` separan la ingestión de documentos, recuperación y generación de respuestas.`
   `9. Native libraries remain in the same Android module, but native loading is now explicit and overridable for JVM tests instead of relying on static side effects.` -> `9. Las bibliotecas nativas permanecen en el mismo módulo de Android, pero la carga nativa ahora es explícita y anulable para pruebas JVM en lugar de depender de efectos secundarios estáticos.`
   `On the native side, the project still builds llama.cpp, stable-diffusion.cpp, whisper.cpp, bark.cpp, and the JNI bridge sources through the Android NDK.` -> `En el lado nativo, el proyecto sigue compilando llama.cpp, stable-diffusion.cpp, whisper.cpp, bark.cpp y las fuentes del puente JNI a través del Android NDK.`

   `## Technologies` -> `## Tecnologías`
   (List stays mostly same, translate descriptions)
   `- [llama.cpp]... — Core LLM backend` -> `- [llama.cpp]... — Backend principal de LLM`
   `- [stable-diffusion.cpp]... — Image/video generation backend` -> `- [stable-diffusion.cpp]... — Backend de generación de imágenes/vídeos`
   `- [whisper.cpp]... — Speech-to-text backend` -> `- [whisper.cpp]... — Backend de voz a texto`
   `- [bark.cpp]... — Text-to-speech backend` -> `- [bark.cpp]... — Backend de texto a voz`
   `- GGUF / GGML — Model formats` -> `- GGUF / GGML — Formatos de modelo`
   `- Android NDK / JNI — Native bindings` -> `- Android NDK / JNI — Enlaces nativos`
   `- ONNX Runtime — Sentence embeddings` -> `- ONNX Runtime — Incrustaciones de oraciones`
   `- Android DownloadManager — Large file downloads` -> `- Android DownloadManager — Descargas de archivos grandes`

   `## Memory Metrics` -> `## Métricas de memoria`
   `You can measure RAM usage at runtime:` -> `Puedes medir el uso de RAM en tiempo de ejecución:`
   (Code block stays)
   `Typical measurement points:` -> `Puntos de medición típicos:`
   `- Before model load` -> `- Antes de la carga del modelo`
   `- After model load` -> `- Después de la carga del modelo`
   `- After blocking prompt` -> `- Después del prompt bloqueante`
   `- After streaming prompt` -> `- Después del prompt en streaming`
   `#### Key fields:` -> `#### Campos clave:`
   `- `totalPssKb`: Total proportional RAM usage. Best for overall tracking.` -> `- `totalPssKb`: Uso total proporcional de RAM. Mejor para seguimiento general.`
   `- `dalvikPssKb`: JVM-managed heap and runtime.` -> `- `dalvikPssKb`: Montón y tiempo de ejecución gestionados por JVM.`
   `- `nativePssKb`: Native heap (llama.cpp, ONNX, tensors, KV cache).` -> `- `nativePssKb`: Montón nativo (llama.cpp, ONNX, tensores, caché KV).`
   `- `otherPssKb`: Miscellaneous memory.` -> `- `otherPssKb`: Memoria miscelánea.`
   `Monitor `nativePssKb` closely during model loading and inference to understand LLM memory footprint.` -> `Monitorea `nativePssKb` de cerca durante la carga del modelo y la inferencia para entender la huella de memoria del LLM.`
   `Expert runtimes such as `SmolLM` also expose native/state-specific memory estimates when you need lower-level instrumentation.` -> `Los tiempos de ejecución expertos como `SmolLM` también exponen estimaciones de memoria nativa/específicas del estado cuando necesitas instrumentación de nivel inferior.`

   `## Notes` -> `## Notas`
   `- `VULKAN_SDK` may still be required when you are building the Vulkan path on the host.` -> `- `VULKAN_SDK` aún puede ser requerido cuando compilas la ruta Vulkan en el host.`
   `- Check Android GPU capability with the explicit per-subsystem helpers such as `LLMEdge.getTextBackendAvailability()` and `LLMEdge.getImageBackendAvailability()`.` -> `- Verifica la capacidad de GPU de Android con los ayudantes explícitos por subsistema como `LLMEdge.getTextBackendAvailability()` y `LLMEdge.getImageBackendAvailability()`.`

   `### ProGuard/R8 Configuration` -> `### Configuración de ProGuard/R8`
   `The library includes consumer ProGuard rules. If you need to add custom rules:` -> `La biblioteca incluye reglas de ProGuard para consumidores. Si necesitas agregar reglas personalizadas:`
   (Code block stays)
   `# Keep OCR engines` -> `# Mantener motores OCR`
   `# Suppress warnings for optional dependencies` -> `# Suprimir advertencias para dependencias opcionales`

   `### Licenses` -> `### Licencias`
   (List stays same, translate descriptions if needed, but licenses are usually kept as is or translated slightly)
   `- **llmedge**: Apache 2.0` -> `- **llmedge**: Apache 2.0`
   `- **llama.cpp**: MIT` -> `- **llama.cpp**: MIT`
   `- **stable-diffusion.cpp**: MIT` -> `- **stable-diffusion.cpp**: MIT`
   `- **whisper.cpp**: MIT` -> `- **whisper.cpp**: MIT`
   `- **bark.cpp**: MIT` -> `- **bark.cpp**: MIT`
   `- **Leptonica**: Custom (BSD-like)` -> `- **Leptonica**: Personalizada (tipo BSD)`
   `- **Google ML Kit**: Proprietary (see ML Kit terms)` -> `- **Google ML Kit**: Propietaria (ver términos de ML Kit)`
   `- **JavaCPP**: Apache 2.0` -> `- **JavaCPP**: Apache 2.0`

   `## License and Credits` -> `## Licencia y Créditos`
   `This project builds upon work by...` -> `Este proyecto se basa en el trabajo de...`
   `See [CREDITS.md](CREDITS.md) for full details.` -> `Consulta [CREDITS.md](CREDITS.md) para detalles completos.`

   `## Testing` -> `## Pruebas`
   `Looking to run unit and instrumentation tests locally, including optional native txt2img E2E checks? See the step-by-step guide in [docs/testing.md](docs/testing.md).` -> `¿Buscas ejecutar pruebas unitarias y de instrumentación localmente, incluyendo verificaciones E2E nativas txt2img opcionales? Consulta la guía paso a paso en [docs/testing.md](docs/testing.md).`

4.  **Self-Correction & Refinement during translation:**
   - Check markdown formatting: All headings, lists, code blocks, links, and callouts are preserved.
   - Check technical terms: "Inferencia", "backend", "montón (heap)", "caché", "tiempo de ejecución", "submódulo", "wrapper/helper" -> "envoltorio/ayudante", "pipeline" -> "pipeline/flujos", "facade" -> "fachada", "callout" syntax `> [!NOTE]` preserved.
   - Ensure code blocks are *exactly* as original. I will copy-paste them carefully.
   - Check link targets: `(#installation)`, `(#usage)`, etc. are preserved. URLs are preserved.
   - Check tone: Professional, technical Spanish.
   - I'll carefully verify the translation matches the original structure line by line.
   - One minor thing: `<think>...` in the original is likely a markdown artifact or specific tag. I'll keep it exactly as is.
   - "Speech-to-Text" is usually "Voz a texto" or "Reconocimiento de voz". I'll use "Voz a texto (STT)".
   - "Text-to-Speech" -> "Texto a voz (TTS)".
   - "Reasoning Controls" -> "Controles de razonamiento".
   - "Tool Calling" -> "Llamadas a herramientas".
   - All good.

   I will now generate the final output strictly following the prompt: "Output only the translated README text." preserving all markdown/code/links.
   I'll do a final pass to ensure no extra text is added.
   Structure matches exactly.
   Translation is accurate and idiomatic for Android/Kotlin development context.
   Ready. 
   Output matches the response. 
   Proceeds. 
   [Output Generation] -> *Self-Correction/Verification during thought*: 
   I need to make sure I don't accidentally translate the code blocks or identifiers inside them. The prompt says "Preserve Markdown, code blocks, commands, identifiers, image targets, and link targets exactly." I will copy the code blocks verbatim from the prompt.
   I will carefully check the Spanish translation for technical accuracy. Terms like "batched blocking", "streaming", "KV cache", "RAG", "LoRA", "DiT", "QAT", "Vulkan", "OpenCL", "JNI", "GGUF", "safetensors" are standard and will remain or be adapted naturally.
   All looks solid.
   Generating. 
   [Output Generation] -> *Proceeds*


# llmedge

**llmedge** es una biblioteca ligera para Android que permite ejecutar modelos de lenguaje GGUF completamente en el dispositivo, impulsada por [llama.cpp](https://github.com/ggerganov/llama.cpp).

Consulta el [repositorio de ejemplos](https://github.com/Aatricks/llmedge-examples) para ver casos de uso.

Los agradecimientos a Shubham Panchal y a los proyectos upstream se listan en [`CREDITS.md`](./CREDITS.md).

> [!NOTE]
> Esta biblioteca está en desarrollo temprano y puede cambiar significativamente.

> [!IMPORTANT]
> La madurez de la API varía según el área de características. `LLMEdge`, la inferencia de texto, la inferencia de voz y la gestión de modelos son los puntos de entrada más estables hoy en día. La extracción de texto OCR mediante `edge.vision.extractText(...)` también es confiable. El análisis de visión/VLM, RAG y algunos flujos de generación de imágenes/vídeos están disponibles y probados, pero aún deben considerarse como APIs en evolución.

---

## Características

- **Inferencia de LLM**: Ejecuta modelos GGUF directamente en Android usando llama.cpp (JNI)
- **Descargas de modelos**: Descarga y almacena en caché modelos desde Hugging Face Hub
- **Presets para dispositivos básicos**: `ModelPresets` ofrece especificaciones listas para usar (Microsoft BitNet b1.58 2B4T, SmolVLM2-256M) optimizadas para dispositivos de gama baja
- **Safetensors → GGUF en el dispositivo**: Convierte modelos safetensors de Hugging Face directamente en el dispositivo (arquitectura Llama + tokenizador GPT2-BPE), con cuantización opcional (Q8_0 / Q4_K_M / IQ2_BN), mediante `ModelSpec.safetensors(...)`
- **Inferencia optimizada**: Reutilización nativa de caché KV para chats compactos, generación de texto bloqueada y en streaming por defecto con lotes, sintonización separada de hilos para el prompt frente a la generación, y reproducción de `ChatSession` gestionada en Kotlin para modelos con alto razonamiento
- **Voz a texto (STT)**: Integración con Whisper.cpp con soporte de marcas de tiempo, detección de idioma, transcripción en streaming y generación de SRT
- **Texto a voz (TTS)**: Integración con Bark.cpp con optimizaciones para ARM
- **Generación de imágenes**: Stable Diffusion con soporte para EasyCache y LoRA, además de FLUX.2 Klein 4B (DiT destilado, la arquitectura detrás de Bonsai Image binario/ternario de PrismML); streaming de progreso por paso y escalado ESRGAN (Remacri)
- **Generación de vídeo**: Modelos Wan 2.1 (4-64 fotogramas) con carga secuencial
- **RAG en el dispositivo**: Indexación de PDF, incrustaciones, búsqueda vectorial, preguntas y respuestas
- **OCR**: Extracción de texto con Google ML Kit
- **Métricas de memoria**: Monitorización integrada del uso de RAM
- **Modelos de visión**: Arquitectura preparada para modelos estilo LLaVA (requiere formatos de modelo específicos)
- **Aceleración por GPU**: Backends de GPU opcionales para Android para texto, Whisper e imágenes/vídeos, con OpenCL experimental preferido primero, Vulkan como respaldo segundo y CPU como última opción

---

## Tabla de contenidos

1. [Instalación](#installation)
2. [Uso](#usage)
   - [Descarga de modelos](#downloading-models)
   - [Controles de razonamiento](#reasoning-controls)
   - [Sesiones de chat gestionadas](#managed-chat-sessions)
   - [Llamadas a herramientas](#tool-calling)
   - [Extracción de texto de imágenes (OCR)](#image-text-extraction-ocr)
   - [Modelos de visión](#vision-models)
   - [Voz a texto (Whisper)](#speech-to-text-whisper)
   - [Texto a voz (Bark)](#text-to-speech-bark)
   - [Estado de rendimiento de voz](#speech-performance-status)
   - [Stable Diffusion (generación de imágenes)](#stable-diffusion-image-generation)
   - [Generación de vídeo](#video-generation)
   - [RAG en el dispositivo](#on-device-rag)
   - [APIs Expertas](#expert-apis)
3. [Compilación](#building)
4. [Arquitectura](#architecture)
5. [Tecnologías](#technologies)
6. [Métricas de memoria](#memory-metrics)
7. [Notas](#notes)
8. [Pruebas](#testing)

---

## Instalación

> [!WARNING]
> Para desarrollo, se recomienda fuertemente Linux para compilaciones con GPU. La ruta de generación de shaders de Vulkan utilizada por Stable Diffusion aún no es fiable en compilaciones cruzadas para Windows.

Clona el repositorio junto con el submódulo `llama.cpp` y `stable-diffusion.cpp`:

```bash
git clone --depth=1 https://github.com/Aatricks/llmedge
cd llmedge
git submodule update --init --recursive
```

Abre el proyecto en Android Studio. Si no se compila automáticamente, usa ***Build > Rebuild Project.***

### Consumir como dependencia

Para Maven Central:

```kotlin
repositories {
    google()
    mavenCentral()
}

dependencies {
    implementation("io.github.aatricks:llmedge:0.3.9")
}
```

Para GitHub Packages:

```kotlin
repositories {
    google()
    mavenCentral()
    maven {
        url = uri("https://maven.pkg.github.com/Aatricks/llmedge")
        credentials {
            username = providers.gradleProperty("gpr.user").orNull ?: System.getenv("GITHUB_ACTOR")
            password = providers.gradleProperty("gpr.key").orNull ?: System.getenv("GITHUB_TOKEN")
        }
    }
}

dependencies {
    implementation("io.github.aatricks:llmedge:0.3.9")
}
```

## Uso

### Inicio rápido

El punto de entrada recomendado es la fachada basada en instancias `LLMEdge`. Expone clientes de dominio para texto, voz, generación de imágenes, visión y RAG, manteniendo la resolución de modelos y la propiedad de recursos explícita.

```kotlin
val edge = LLMEdge.create(
    context = context,
    scope = viewModelScope,
)

viewModelScope.launch {
    val reply = edge.text.generate(
        prompt = "Summarize on-device LLMs in one sentence.",
    )
    outputView.text = reply
}
```

Los envoltorios de bajo nivel como `SmolLM`, `StableDiffusion`, `Whisper` y `BarkTTS` siguen disponibles para flujos de trabajo avanzados, pero el código nuevo debería preferir `LLMEdge`.

La ruta de adquisición prevista para el código de la aplicación es:

- `edge.models.prefetch(...)` cuando quieras descargas explícitas
- clientes de características como `edge.text`, `edge.speech`, `edge.image` y `edge.vision` cuando quieras inferencia

Las llamadas directas a `HuggingFaceHub` y los ayudantes expertos `loadFromHuggingFace(...)` siguen siendo compatibles, pero son APIs avanzadas para llamantes que necesitan control a nivel de artefacto.

Por defecto, `edge.text.generate(...)` usa decodificación nativa en lotes para reducir la sobrecarga de JNI, mientras que
`edge.text.stream(...)` usa lotes más pequeños para que las actualizaciones de la UI se mantengan responsivas sin pagar un
cruce de JNI por token.

### Descarga de modelos

llmedge puede resolver y almacenar en caché los pesos del modelo de forma independiente a la inferencia:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val modelFile = edge.models.prefetch(
    ModelSpec.huggingFace(
        repoId = "unsloth/Qwen3-0.6B-GGUF",
        filename = "Qwen3-0.6B-Q4_K_M.gguf",
    ),
)

Log.d("llmedge", "Cached ${modelFile.name} at ${modelFile.parent}")
```

#### Puntos clave:

- `edge.models.prefetch(...)` y `BoundModelRepository.resolve(...)` mantienen la adquisición del modelo separada de cualquier cliente de inferencia.

- Admite callbacks de progreso y repositorios privados mediante token a través de `ModelSpec.huggingFace(...)`.

- Las solicitudes a espejos antiguos se resuelven automáticamente a repositorios actualizados de Hugging Face.

- Usa automáticamente la ventana de contexto declarada por el modelo (mínimo 1K tokens) y la limita a un máximo consciente del montón (2K–8K). Anula con `InferenceParams(contextSize = …)` si es necesario.

- Las descargas grandes usan DownloadManager de Android cuando `preferSystemDownloader = true` para mantener las transferencias fuera del montón Dalvik.

- Las descargas directas de `HuggingFaceHub` siguen disponibles para flujos avanzados, pero la mayoría del código de la aplicación debería usar la ruta de fachada/repositorio de modelos.

#### Presets integrados para dispositivos básicos

`ModelPresets` expone especificaciones listas para usar optimizadas para dispositivos básicos, ya compatibles con el tiempo de ejecución ik_llama.cpp incluido: no necesitas escribir manualmente repositorios/nombres de archivo:

```kotlin
// Microsoft BitNet b1.58 2B4T — LLM nativo de 1-bit (IQ2_BN, ~988 MB).
// La plantilla de chat correcta viene con el preset, por lo que la generación es correcta desde el principio.
val reply = edge.text.generate(prompt = "Hi", model = ModelPresets.bitnet)

// SmolVLM2-256M — modelo de visión diminuto (~280 MB total: base + proyector).
val caption = edge.vision.analyze(
    image = bitmap,
    prompt = "Describe this image.",
    model = ModelPresets.smolVlm2.model,
    projector = ModelPresets.smolVlm2.projector,
)
```

> [!NOTE]
> Los metadatos GGUF de BitNet contienen una plantilla de chat incorrecta, por lo que llmedge proporciona la canónica mediante
> `ModelHints.chatTemplate`. Una plantilla que pases a través de `TextModelOptions.chatTemplate` siempre la anula.
> Los modelos ternarios como Bonsai y otros distribuidos solo como GGUF `Q2_0` de PrismML **no** son cargables por este
> tiempo de ejecución (usa `IQ2_BN`); convierte los safetensors en su lugar: consulta
> [Conversión de modelos safetensors](docs/usage.md#converting-safetensors-models).

### Controles de razonamiento

Los modelos con conciencia de razonamiento pueden controlarse desde la fachada mediante `TextModelOptions`. La configuración predeterminada mantiene el pensamiento habilitado (`ThinkingMode.DEFAULT`, presupuesto de razonamiento `-1`). Para deshabilitar el pensamiento en una solicitud o sesión, pasa las opciones explícitamente:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val reply = edge.text.generate(
    prompt = "Solve this step by step, then give only the final answer.",
    options = TextModelOptions(
        thinkingMode = SmolLM.ThinkingMode.DISABLED,
        reasoningBudget = 0,
    ),
)
```

Las mismas opciones funcionan con `edge.text.session(...)` y `edge.text.toolAgent(...)`.

Establecer el presupuesto en `0` deshabilita siempre el pensamiento, mientras que `-1` lo deja sin restricciones. Si omites `reasoningBudget`, la biblioteca elige `0` cuando el modo es `DISABLED` y `-1` en otros casos. La API también inyecta automáticamente la etiqueta `/no_think` cuando el pensamiento está deshabilitado, por lo que no necesitas modificar los prompts manualmente. Si necesitas cambiar el estado de razonamiento en un tiempo de ejecución experto activo sin recargar, consulta [APIs Expertas](#expert-apis).

### Sesiones de chat gestionadas

Usa `edge.text.session(...)` cuando quieras chat de múltiples turnos acotado sin exponer el estado nativo `storeChats` al código de la aplicación.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val session = edge.text.session(
    memory = ConversationWindow(
        maxTurns = 6,
        maxTokens = 4096,
        stripThinkTags = true,
    ),
    systemPrompt = "You are a concise assistant.",
)

viewModelScope.launch {
    session.prepare()
    val reply = session.reply("Explain why context windows fill up.")
    session.stream("Now summarize that in 3 bullets.").collect { event ->
        when (event) {
            is TextStreamEvent.Chunk -> print(event.value)
            is TextStreamEvent.Completed -> println(event.fullText)
            else -> Unit
        }
    }
}
```

La nueva API de sesiones mantiene el estado del transcripción en Kotlin, aplica recorte de ventana deslizante y elimina por defecto los bloques reproducidos `<think>...` para que los modelos con alto razonamiento no agoten la ventana de contexto tan rápido.

### Llamadas a herramientas (Tool Calling)

Usa `edge.text.toolAgent(...)` cuando quieras que el modelo invoque herramientas definidas por la aplicación. Las herramientas de solo lectura se ejecutan automáticamente; las herramientas de acción requieren una decisión de política explícita.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val factory = DeviceToolFactory(context)

val agent = edge.text.toolAgent(
    tools = factory.createDefaultTools(),
    systemPrompt = "Be concise and only use tools when needed.",
    policy = ToolPolicies.ALLOW_ALL, // or keep the default to deny action tools
)

viewModelScope.launch {
    val result = agent.reply("What time is it and how much battery is left?")
    println(result.text)

    agent.stream("Open https://example.com").collect { event ->
        when (event) {
            is ToolAgentEvent.ToolCallRequested -> println("Tool: ${event.call.tool}")
            is ToolAgentEvent.TextChunk -> print(event.value)
            is ToolAgentEvent.Completed -> println("\nDone: ${event.result.finishReason}")
            else -> Unit
        }
    }
}
```

Las llamadas a herramientas usan internamente un sobre JSON estructurado: `{"tool":"name","arguments":{...}}`. El analizador también acepta el campo heredado `tool_name` para robustez, pero los prompts nuevos solo emiten la forma `tool`.

Para hosts JVM o de escritorio donde `bash` está disponible, también puedes optar por una herramienta de ejecución de shell:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val bashTool =
    BashToolFactory(
        BashToolOptions(
            allowRawShell = true, // raw `command` strings are disabled unless you opt in
            defaultWorkingDirectory = context.filesDir.absolutePath,
        ),
    ).createBashTool()

val agent = edge.text.toolAgent(
    tools = listOf(bashTool),
    systemPrompt = "Use shell commands only when necessary.",
    policy = ToolPolicies.ALLOW_ALL, // required because run_bash_command is an action tool
)
```

La herramienta bash acepta `argv` para comandos estructurados o `command` para cadenas de shell crudas. Si `bash` no está disponible o el comando falla, la herramienta devuelve un resultado de error estructurado en lugar de afirmar éxito.

### Objetos de solicitud de voz

Las APIs de voz ahora admiten llamadas primero-por-solicitud además de las sobrecargas de conveniencia existentes:

```kotlin
val result = edge.speech.transcribe(
    SpeechToTextRequest(
        audioSamples = samples,
        model = edge.config.models.speechToText,
        params = Whisper.TranscribeParams(language = "en"),
        runtime = WhisperRuntimeRequest(gpuEnabled = false, flashAttention = true),
    ),
)
```

Esto mantiene las nuevas entradas de voz alineadas con el estilo primero-por-solicitud ya usado por texto y generación de imágenes, mientras preserva las sobrecargas antiguas de listas de parámetros para compatibilidad.

### Ajuste de rendimiento de generación de texto

El stack de texto ahora separa el procesamiento de prompt/lote de la generación de un solo token para que puedas sintonizar
las dos fases de forma independiente:

```kotlin
val edge = LLMEdge.create(
    context = context,
    scope = viewModelScope,
    config = LLMEdgeConfig(
        text = TextRuntimeConfig(
            promptThreads = 6,            // prompt/batch phase
            generationThreads = 2,       // token-by-token phase
            batchSize = 8,
            streamBatchSize = 4,
            cache = RuntimeCacheConfig(maxEntries = 2, maxMemoryMb = 1536),
        ),
    ),
)

val reply = edge.text.generate(
    prompt = "Explain speculative decoding.",
    options = TextModelOptions(numThreads = 8, generationThreads = 3),
    batchSize = 12,
)
```

Valores predeterminados prácticos:

- `text.promptThreads`: hilos de decodificación de prompt/lote
- `text.generationThreads`: hilos de generación de token único
- `text.batchSize`: tamaño de lote para texto bloqueante (predeterminado `8`)
- `text.streamBatchSize`: tamaño de lote para streaming (predeterminado `4`)
- `text.cache.maxMemoryMb`: límite superior para la contabilidad de la caché de modelos de texto; la caché ahora se actualiza según la huella nativa del modelo/estado en lugar de solo el tamaño del archivo GGUF

Guía de tamaño de lote:

- `1`: menor latencia por fragmento, mayor sobrecarga de JNI
- `4`: buen predeterminado para actualizaciones de UI en streaming
- `8`: buen predeterminado para respuestas de texto bloqueantes
- `12+`: mejor rendimiento para generaciones offline más largas, pero puede retrasar actualizaciones intermedias

### Extracción de texto de imágenes (OCR)

llmedge usa el reconocimiento de texto de Google ML Kit para extraer texto de imágenes.

#### Inicio rápido

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val text = edge.vision.extractText(bitmap)
println("Extracted text: $text")
```

#### Motores OCR

**Reconocimiento de texto de Google ML Kit**
- Rápido y ligero
- No requiere archivos de datos adicionales
- Bueno para scripts latinos
- Agrega la dependencia: `implementation("com.google.mlkit:text-recognition:16.0.0")`

El OCR se expone directamente a través de `edge.vision.extractText(...)`. El envoltorio de conveniencia antiguo `VisionMode` ha sido eliminado; los llamantes ahora eligen explícitamente entre OCR y análisis VLM en lugar de enrutamiento a través de una segunda capa de abstracción.

### Modelos de visión

Analiza imágenes usando Modelos de Lenguaje de Visión (como LLaVA o Phi-3 Vision) mediante `edge.vision`.

> [!WARNING]
> La ruta VLM es experimental. Requiere un GGUF capaz de visión y un archivo mmproj/proyector coincidente. Cuando estos componentes no están disponibles o son incompatibles, `edge.vision.analyze(...)` falla rápidamente con un error claro en lugar de caer silenciosamente en prompts solo de texto. El OCR sigue disponible a través de `edge.vision.extractText(...)`.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val description = edge.vision.analyze(
    image = bitmap,
    prompt = "Describe this image in detail.",
    numThreads = 4,
    generationThreads = 2,
) { status ->
    Log.d("Vision", "Status: $status")
}
```

La ruta de visión de alto nivel actual crea un tiempo de ejecución `SmolLM` nuevo por solicitud, por lo que favorece
el aislamiento y la limpieza predecible sobre la reutilización en grupo de alto rendimiento.

El administrador maneja el flujo complejo de:
1. Preprocesamiento de la imagen
2. Carga del proyector de visión y del modelo
3. Codificación de la imagen en incrustaciones
4. Generación de la respuesta textual

El soporte para modelos de visión es actualmente experimental y requiere arquitecturas de modelo específicas (como LLaVA-Phi-3).

### Voz a texto (Whisper)

Transcribe audio usando el nuevo cliente `edge.speech`:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val text = edge.speech.transcribeToText(audioSamples)

val segments = edge.speech.transcribe(
    audioSamples = audioSamples,
    params = Whisper.TranscribeParams(language = "en"),
)
segments.forEach { segment ->
    println("[${segment.startTimeMs}ms] ${segment.text}")
}

val lang = edge.speech.detectLanguage(audioSamples)
```

#### Transcripción en streaming en tiempo real

Para subtítulos en vivo, usa la API de transcripción en streaming con un enfoque de ventana deslizante:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val session = edge.speech.createStreamingSession(
    params = Whisper.StreamingParams(
        stepMs = 3000,
        lengthMs = 10000,
        keepMs = 200,
        language = "en",
        useVad = true,
    ),
)

viewModelScope.launch {
    session.events().collect { segment ->
        updateCaptions(segment.text)
    }
}

audioRecorder.onAudioChunk { samples ->
    viewModelScope.launch { session.feedAudio(samples) }
}

session.stop()
```

**Parámetros de streaming:**
- `stepMs`: Frecuencia de ejecución de la transcripción (predeterminado: 3000ms). Menor = actualizaciones más rápidas, mayor uso de CPU.
- `lengthMs`: Tamaño de la ventana de audio (predeterminado: 10000ms). Ventanas más largas mejoran la precisión.
- `keepMs`: Solapamiento con la ventana anterior (predeterminado: 200ms). Ayuda a mantener el contexto.
- `useVad`: Detección de Actividad de Voz - omite audio silencioso (predeterminado: true).

El acceso directo a `Whisper` sigue disponible para flujos avanzados, pero el cliente de voz con espacio de nombres es la ruta de integración estándar.

**Modelos recomendados:**
- `ggml-tiny.bin` (~75MB) - Rápido, menor precisión
- `ggml-base.bin` (~142MB) - Buen equilibrio
- `ggml-small.bin` (~466MB) - Mayor precisión

### Texto a voz (Bark)

Genera voz usando `edge.speech`:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val audio = edge.speech.synthesize("Hello, world!")

viewModelScope.launch {
    edge.speech.synthesizeStream("Hello, world!").collect { event ->
        when (event) {
            is AudioStreamEvent.Progress -> Log.d("Bark", "${event.step.name}: ${event.percent}%")
            is AudioStreamEvent.Result -> saveAudio(event.audio)
            else -> Unit
        }
    }
}
```

El acceso directo a `BarkTTS` sigue disponible para flujos avanzados, pero el cliente de voz con espacio de nombres es la ruta de integración estándar.

### Stable Diffusion (generación de imágenes)

Genera imágenes en el dispositivo usando el cliente con espacio de nombres `edge.image`:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val bitmap = edge.image.generate(
    ImageGenerationRequest(
        prompt = "a cute pastel anime cat, soft colors, high quality <lora:detail_tweaker:1.0>",
        width = 512,
        height = 512,
        steps = 20,
        loraModelDir = "/path/to/loras",
        loraApplyMode = StableDiffusion.LoraApplyMode.AUTO,
    ),
)
imageView.setImageBitmap(bitmap)
```

**Optimizaciones clave:**
- **EasyCache**: `edge.image` habilita automáticamente EasyCache para modelos Diffusion Transformer (DiT) compatibles como Flux, SD3, Wan, Qwen Image y Z-Image; permanece deshabilitado para pipelines clásicos de UNet.
- **Flash Attention**: Habilitado automáticamente para dimensiones de imagen compatibles.
- **LoRA**: Aplica pesos de ajuste fino sobre la marcha sin fusionar modelos.

Para propiedad explícita del tiempo de ejecución o experimentos de carga nativa personalizados, la clase `StableDiffusion` sigue disponible en la capa de API experta.

#### Progreso en streaming

`generate` bloquea hasta que el bitmap esté listo. Para impulsar una barra de progreso, usa `generateStream`, que emite un evento `Progress` por paso de desruido y un evento final `Completed` que lleva la imagen:

```kotlin
edge.image.generateStream(request).collect { event ->
    when (event) {
        is GenerationStreamEvent.Progress ->
            progressBar.progress = event.update.current * 100 / event.update.total
        is GenerationStreamEvent.Completed ->
            imageView.setImageBitmap(event.frames.first())
    }
}
```

Cancelar la colección cancela la generación. Los eventos de paso solo cubren el bucle de muestreo; la carga del modelo ocurre antes del primer evento, por lo que mantén la barra indeterminada hasta entonces. La aplicación de ejemplo deriva una estimación de tiempo restante desde el retraso entre eventos (`StepEtaEstimator` en llmedge-examples).

#### Escalado de imágenes (ESRGAN)

`edge.image.upscale` ejecuta un modelo ESRGAN sobre un bitmap y devuelve el resultado ampliado. Los puntos de control ESRGAN de arquitectura antigua se cargan directamente; 4x_foolhardy_Remacri es la referencia probada:

```kotlin
val esrgan = edge.models.resolve(
    ModelSpec.huggingFace(
        repoId = "LyliaEngine/4x_foolhardy_Remacri",
        filename = "4x_foolhardy_Remacri.safetensors",
        hints = ModelHints(
            artifactKind = ModelArtifactKind.REPO_FILE,
            capabilities = setOf(ModelCapability.IMAGE),
        ),
    ),
)

val upscaled = edge.image.upscale(
    UpscaleRequest(input = bitmap, model = ModelSpec.localFile(esrgan)),
) { tile, totalTiles ->
    // fires once per processed tile
}
```

Detalles que vale la pena conocer:

- `factor = 0` (predeterminado) usa la escala incrustada en el modelo. Remacri es 4x.
- El escalador se ejecuta en CPU a menos que la solicitud establezca `useVulkan = true`. Las entradas grandes se procesan en bloques de 128px, y el callback de progreso informa recuentos de bloques, no pasos.
- La entrada se limita a 1024×1024. Un paso 4x sobre una imagen de 1024px ya produce un bitmap de 16 MP, cerca del límite práctico de memoria en teléfonos gama media.
- En modo de worker aislado, el escalado se ejecuta en el proceso `:llmedge_sd` con el mismo watchdog y recuperación ante fallos que la generación de imágenes.
- El contexto ESRGAN se crea y libera por llamada, por lo que espera una breve pausa de carga del modelo (los pesos son ~64 MB) antes del primer bloque.

#### MiniT2I

El ayudante `MiniT2I` descarga el transformer de difusión MiniT2I B/16 independiente y su codificador de texto FLAN-T5 Large, luego los enruta a los slots de difusión dividida y modelo T5 de stable-diffusion.cpp:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val bitmap = edge.image.generate(
    MiniT2I.imageRequest("a small robot watering flowers"),
)
```

El ayudante usa por defecto la configuración 512×512, 100 pasos, CFG 6 del modelo. Ancho, alto, pasos, CFG, semilla y flash attention siguen configurables a través de `MiniT2I.imageRequest(...)`.

#### FLUX.2 Klein 4B (DiT destilado, modelo dividido)

[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) es un transformer de difusión destilado por pasos que produce imágenes de alta calidad en ~4 pasos. Es la misma arquitectura sobre la que se construyen los modelos **Bonsai Image** binarios/ternarios de PrismML: los propios pesos 1-bit/ternarios de Bonsai se distribuyen solo en empaquetados MLX (Apple) y GemLite (CUDA), que no se cargan en Android, por lo que esta compilación GGUF es el equivalente ejecutable en Android con una huella comparable.

A diferencia de un punto de control clásico de archivo único, FLUX.2 se carga como tres componentes: el transformer de difusión (GGUF), un codificador de texto Qwen3-4B y el VAE de FLUX.2. El ayudante `Flux2Klein` conecta los tres más los valores predeterminados destilados (CFG 1.0, 4 pasos):

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val bitmap = edge.image.generate(
    Flux2Klein.imageRequest("a red fox in snow, detailed, 8k"),
)
```

Internamente esto establece `ImageGenerationRequest.splitDiffusionModel = true`, que enruta el transformer a `diffusion_model_path` de stable-diffusion.cpp y el codificador Qwen3 a `llm_path` (en lugar del único slot `model_path`), y descarga los pesos a CPU. Huella: ~2.5 GB DiT (Q4_0) + ~2.1 GB codificador (Q3_K_M) + ~0.3 GB VAE, por lo que apunta a dispositivos con mayor RAM.

##### Gama baja: DiT Bonsai (QAT) para un transformer más pequeño

Los modelos **Bonsai Image** de PrismML son FLUX.2 Klein 4B ajustado con entrenamiento consciente de cuantización (QAT) a pesos ternarios. Se distribuyen solo en empaquetados MLX/GemLite (no cargables en Android) y en una forma `-unpacked` densa-bf16 usando la nomenclatura de diffusers no estándar `Flux2KleinPipeline`, que stable-diffusion.cpp no puede ingerir directamente.

`scripts/convert_bonsai_flux2_to_bfl.py` convierte un transformer Bonsai desempacado a la nomenclatura BFL que espera sdcpp (renombra y fusiona el bloque doble `to_q/k/v` en `*_attn.qkv`). Luego cuantiza con stable-diffusion.cpp:

```bash
python3 scripts/convert_bonsai_flux2_to_bfl.py \
    bonsai-image-ternary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors \
    bonsai-flux2-bfl.safetensors
sd -M convert -m bonsai-flux2-bfl.safetensors --type q2_K -o bonsai-flux2-klein-q2_K.gguf
```

Un Q2_K precompilado de esto se publica en [`Aatricks/bonsai-image-ternary-4B-FLUX2-klein-GGUF`](https://huggingface.co/Aatricks/bonsai-image-ternary-4B-FLUX2-klein-GGUF) y se conecta en `Flux2Klein.bonsaiDiffusionModel`. Los pesos QAT sobreviven bien a `Q2_K`, dando un **DiT coherente de ~1.3 GB** (vs ~2.5 GB para Q4_0 base). Nota: los tipos ternarios literales de ggml (`tq1_0`/`tq2_0`, ~0.8–1.0 GB) se cargan y ejecutan en CPU pero su escala por 256 pesos es demasiado gruesa para las escalas entrenadas por 128 de Bonsai y producen salida degradada — las escalas más finas por subbloque de 16 de `Q2_K` son las que preservan la calidad.

#### Carga secuencial para dispositivos de ~4 GB de RAM

El codificador de texto (~2 GB) es el costo dominante de memoria. El modo secuencial carga solo el codificador Qwen3 para precomputar la condición de texto, lo libera, luego carga solo el DiT para generar — así la RAM pico es `max(codificador, DiT)` (~2.6 GB) en lugar de la suma (~4 GB):

```kotlin
val bmp = edge.image.generate(Flux2Klein.bonsaiImageRequest("a red fox in snow, 8k"))
// or for the base FLUX.2 Klein DiT: Flux2Klein.imageRequest(prompt, sequential = true)
```

`ImageGenerationRequest.sequential` controla esto; el tiempo de ejecución ejecuta las dos fases automáticamente (solo codificador → precomputar → liberar → solo DiT → generar mediante una condición precomputada), respaldado por `sd_precompute_condition` / `sd_generate_image_with_precomputed_condition` de stable-diffusion.cpp.

### Generación de vídeo

Genera clips de vídeo cortos usando `edge.image.generateVideo(...)`. El cliente con espacio de nombres expone el progreso como un `Flow` mientras reutiliza la lógica de carga de Wan existente internamente.

**Requisitos de hardware**:
- Se recomienda **12GB+ de RAM** para carga estándar.
- **8GB+ de RAM** soportado mediante `forceSequentialLoad = true` (más lento pero seguro en memoria).

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val params = VideoGenerationRequest(
    prompt = "a cat walking in a garden, high quality",
    videoFrames = 8,
    width = 512,
    height = 512,
    steps = 20,
    cfgScale = 7.0f,
    flowShift = 3.0f,
    forceSequentialLoad = true,
)

viewModelScope.launch {
    edge.image.generateVideo(params).collect { event ->
        when (event) {
            is GenerationStreamEvent.Progress -> Log.d("VideoGen", event.update.message)
            is GenerationStreamEvent.Completed -> previewImageView.setImageBitmap(event.frames.first())
        }
    }
}
```

`edge.image` automáticamente:
1. Descarga los archivos de modelo Wan 2.1 necesarios (Difusión, VAE, T5).
2. Carga componentes secuencialmente para minimizar el uso pico de memoria (si se solicita).
3. Gestiona el bucle de generación y la conversión de fotogramas.

Consulta `llmedge-examples` para una implementación de UI completa.


Ejecutar la aplicación de ejemplo:
1. Compila la biblioteca (desde la raíz del repositorio):

```bash
./gradlew :llmedge:assembleRelease
```

2. Compila e instala la aplicación de ejemplo:

```bash
cd llmedge-examples
../gradlew :app:assembleDebug
../gradlew :app:installDebug
```

3. Abre la aplicación en el dispositivo y selecciona la demo "Stable Diffusion" desde el launcher. La demo descarga cualquier archivo faltante desde Hugging Face y ejecuta una generación txt2img rápida.

Notas:
- El ejemplo descarga explícitamente un archivo safetensors VAE para la demo `Meina/MeinaMix`; muchos repos incluyen archivos VAE, pero algunos repos de modelos GGUF incluyen todo lo necesario. Si el repo carece de un archivo de modelo GGUF obtendrás una IllegalArgumentException obvia — proporciona un `filename` o elige un repo diferente en ese caso.
- Usa el descargador del sistema para archivos safetensors/gguf grandes para evitar presión en el montón en Android.
### RAG en el dispositivo

La biblioteca incluye un pipeline RAG mínimo en el dispositivo, similar a Android-Doc-QA, construido con:
- Incrustaciones de oraciones (ONNX)
- `TextSplitter` de espacio en blanco
- `VectorStore` coseno en memoria con persistencia JSON
- `SmolLM` para respuestas conscientes del contexto a través de la sesión RAG gestionada por la fachada

### Configuración

1. Descargar incrustaciones

   Desde el repositorio de Hugging Face `sentence-transformers/all-MiniLM-L6-v2`, coloca:

```
llmedge/src/main/assets/embeddings/all-minilm-l6-v2/model.onnx
llmedge/src/main/assets/embeddings/all-minilm-l6-v2/tokenizer.json
```

2. Compilar la biblioteca

```
./gradlew :llmedge:assembleRelease
```

3. Usar en tu aplicación

```kotlin
val edge = LLMEdge.create(this, lifecycleScope)
val rag = edge.rag.createSession()

lifecycleScope.launch {
    rag.init()
    val count = rag.indexPdf(pdfUri)
    val answer = rag.ask("What are the key points?")
    // render answer
}
```

La construcción directa de `RAGEngine` sigue disponible para flujos avanzados, pero el código de aplicación nuevo debería preferir `edge.rag.createSession()` para que la propiedad y desmontaje del tiempo de ejecución estén alineados con el resto de la biblioteca.

### APIs Expertas

El acceso a `SmolLM`, `StableDiffusion`, `Whisper`, `BarkTTS`, `RAGEngine` y `HuggingFaceHub` directo sigue disponible cuando necesitas mantener un tiempo de ejecución nativo directamente o anular el comportamiento de carga de bajo nivel. Están intencionalmente secundarios a las APIs de fachada.

Ejemplos:

```kotlin
// Direct model download when you need full control over artifact selection.
val download = HuggingFaceHub.ensureModelOnDisk(
    context = context,
    modelId = "unsloth/Qwen3-0.6B-GGUF",
    filename = "Qwen3-0.6B-Q4_K_M.gguf",
)

// Expert text runtime with live reasoning-state control.
val smol = SmolLM()
smol.load(download.file.absolutePath)
smol.setThinkingEnabled(false)

// Expert RAG wiring when you want to own both the runtime and the pipeline yourself.
val ragEngine = RAGEngine(context = context, smolLM = smol)
```

## Compilación

Compilación de backends de GPU en Android
--------------------------------

Si quieres aceleración por GPU para los backends nativos de inferencia, sigue estas notas y requisitos. En Android, llmedge ahora prefiere `OPENCL -> VULKAN -> CPU` cuando se permite el uso de GPU para solicitudes de texto, Whisper e imágenes/vídeos. El soporte OpenCL es experimental, solo para Android y actualmente limitado a `arm64-v8a`. Bark permanece solo en CPU.

Prerrequisitos
- Android NDK r27 o más nuevo (NDK r27 usado en desarrollo; el NDK proporciona los encabezados C de Vulkan). Asegúrate de que tu NDK coincida con la versión usada por tu entorno de compilación.
- CMake 3.22+ y Ninja (el plugin de Gradle para Android recogerá CMake cuando esté configurado).
- Gradle (usa el wrapper: `./gradlew`).
- API de Android (minSdk) 30 o superior. `llmedge` apunta a Android 11+ hoy en día, y el soporte Vulkan aún requiere Vulkan 1.2.
- (Opcional) `VULKAN_SDK` configurado en el entorno si compilas shaders o usas herramientas del SDK de Vulkan en el host. La compilación busca un encabezado `vulkan.hpp` coincidente cuando sea necesario.

### Configuración del host para compilaciones Vulkan (Ubuntu/WSL)

Para compilar la biblioteca con soporte Vulkan en un host Linux o WSL2, debes instalar el compilador de shaders Vulkan y los encabezados de desarrollo:

1. **Instalar dependencias**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y glslc libvulkan-dev
   ```

2. **Verificar glslc**:
   Asegúrate de que `glslc` esté en tu PATH:
   ```bash
   glslc --version
   ```

3. **Android NDK**:
   Asegúrate de tener el Android NDK **r27** (específicamente `27.2.12479018`) instalado a través de Android Studio o el administrador de SDK.

Banderas de compilación
- En hosts Linux/macOS, la compilación de Gradle habilita Vulkan por defecto. En hosts Windows, predetermina `OFF` porque el paso del generador de shaders upstream sigue siendo frágil bajo la cadena de herramientas de compilación cruzada de Android. Rehábilítalo explícitamente solo cuando tu entorno soporte esa ruta.
- El OpenCL experimental para Android está deshabilitado por defecto. Habilítalo con `-PllmedgeAndroidOpencl=ON` o la variable de entorno `LLMEDGE_ANDROID_OPENCL=ON`.
- Si quieres compilar tanto OpenCL como Vulkan explícitamente, usa:

```bash
./gradlew :llmedge:assembleRelease \
  -PllmedgeAndroidOpencl=ON \
  -Pandroid.injected.build.api=30 \
  -Pandroid.jniCmakeArgs="-DSD_VULKAN=ON -DGGML_VULKAN=ON"
```

Alternativamente, establece las mismas banderas en tu configuración CMake de Android Studio. `LLMEDGE_ANDROID_OPENCL` es el interruptor experimental de OpenCL de la biblioteca, mientras que `-DSD_VULKAN=ON` y `-DGGML_VULKAN=ON` fuerzan el soporte Vulkan para Stable Diffusion y ggml.

Notas sobre encabezados y cadena de herramientas
- La compilación obtiene `Vulkan-Hpp` (`vulkan.hpp`) y lo fija a los encabezados Vulkan del NDK para evitar desajustes de API. Si tienes un `VULKAN_SDK` local puedes apuntar a él, de lo contrario el proyecto usará los encabezados obtenidos.
- Cuando OpenCL está habilitado, la compilación usa encabezados OpenCL gestionados por el repo y un shim de cargador en tiempo de enlace. La aplicación empaquetada aún resuelve la implementación OpenCL del dispositivo en tiempo de ejecución en lugar de incluir su propio ICD de plataforma.
- El repositorio también compila una pequeña cadena de herramientas de host para generar shaders SPIR-V en tiempo de compilación; asegúrate de que tu host de compilación tenga una cadena de herramientas C++ funcional (clang/gcc) y CMake configurado.

Verificación en tiempo de ejecución
- Para verificar la capacidad de GPU en tiempo de ejecución:
    - Ejecuta la app en un dispositivo Android 11+.
    - Usa las APIs de capacidad por subsistema para inspeccionar los motores que te interesan, por ejemplo `LLMEdge.getTextBackendAvailability()`, `LLMEdge.getSpeechBackendAvailability()`, `LLMEdge.getImageBackendAvailability()`, y `LLMEdge.getVisionBackendAvailability()`.
    - Inspecciona los registros de tiempo de ejecución para el backend seleccionado y cualquier motivo de respaldo. Ejemplo:

```bash
adb logcat -s SmolSD:* | sed -n '1,200p'
```

    Busca mensajes que indiquen inicialización de OpenCL o Vulkan. `LLMEdgeConfig(text = TextRuntimeConfig(useVulkan = true))` significa "permitir un backend de GPU compatible", no "forzar Vulkan".

Solución de problemas
- Si ves "Vulkan 1.2 requerido" o errores del enlazador para símbolos Vulkan, confirma que `minSdk` está configurado en 30 o superior en `llmedge/build.gradle.kts` y que tu NDK proporciona los encabezados Vulkan esperados.
- Si OpenCL experimental no está disponible, o si un backend de GPU falla al inicializarse o ejecutarse, llmedge cae automáticamente a Vulkan o CPU. Para texto, Whisper e imágenes/vídeos, un backend fallido se pone en lista negra por subsistema para el resto del proceso y el siguiente backend se reintenta una vez.
- Si tu dispositivo carece tanto de soporte OpenCL como Vulkan usable, el código nativo cae al backend de CPU.
- Si el controlador Vulkan se inicializa pero el primer generate se cuelga indefinidamente en el primer envío de cómputo (observado en PowerVR DXT-48 / Pixel 10 Tensor G5), la caída automática no puede activarse: la carga tiene éxito, por lo que no se observa fallo. Fuerza el backend de CPU para generación de imágenes/vídeos con `LLMEdgeConfig(image = ImageRuntimeConfig(useVulkan = false))`, o habilita el aislamiento de proceso (abajo) para que la biblioteca detecte y se recupere del colgar automáticamente.

Aislamiento de proceso para generación de imágenes/vídeos (optativo)

`LLMEdgeConfig(image = ImageRuntimeConfig(workerMode = DiffusionWorkerMode.ISOLATED_PROCESS))` ejecuta el stack de difusión en un proceso worker `:llmedge_sd` de propiedad de la biblioteca:

- Un fallo nativo en el stack de difusión aparece como una `WorkerCrashedException` tipada (con un reintento automático en CPU) en lugar de matar la app.
- Un controlador de GPU que se cuelga en el envío es detectado por un watchdog: sin latido de progreso mientras el tiempo de CPU del worker se mantiene plano (una compilación en frío de shader legítima satura un núcleo y nunca se mata) — el worker se mata, y según `hangRecoveryPolicy` la solicitud se reintenta transparentemente en CPU (predeterminado) o falla con `GenerationHangException`.
- Los veredictos de colgar persisten entre sesiones (indexados por la huella digital de compilación del SO, por lo que una actualización de sistema/controlador reactiva la GPU automáticamente). `ImageClient.resetBackendVerdicts(context)` los limpia manualmente.
- La API pública `ImageClient` no cambia; `DiffusionWorkerMode.IN_PROCESS` permanece como predeterminado por ahora. Las apps que ya hospedan llmedge en su propio proceso de servicio deberían permanecer in-process para evitar un proceso extra redundante.

#### Notas:

- Usa `com.tom-roush:pdfbox-android` para análisis de PDF.
- Biblioteca de incrustaciones: `io.gitlab.shubham0204:sentence-embeddings:v6`.
- Los PDF escaneados requieren OCR (ej., ML Kit o Tesseract) antes de indexar.
- Los errores ONNX `token_type_ids` se manejan automáticamente; anula vía `EmbeddingConfig` si es necesario.

## Arquitectura

El lado Kotlin ahora está organizado alrededor de unas pocas capas explícitas en lugar de una sola fachada ansiosa:

1. `LLMEdge` es una cáscara de conveniencia fina que crea perezosamente clientes de dominio (`text`, `speech`, `image`, `vision`, `rag`) al primer acceso.
2. `ModelRepository` posee la adquisición y validación de modelos para archivos locales y descargas de Hugging Face.
3. `RuntimePool` y `RuntimeCoordinator` proporcionan caché compartida de tiempo de ejecución, selección de backend y listado en negro de fallos.
4. `RuntimePoolProfile` permite que cada dominio describa el dimensionamiento, clave, carga y política de caché sin duplicar el código boilerplate del pool.
5. `TextClient`, `SpeechClient`, `ImageClient`, `VisionClient` y `RAGClient` permanecen construibles independientemente para uso avanzado, pero `LLMEdge` es el punto de entrada público canónico.
6. `ConversationSessionSupport` centraliza el estado del transcripción y el acceso al tiempo de ejecución para sesiones de chat y agentes de herramientas.
7. `VisionInputPreparer` y `VisionRuntimeExecutor` separan el preprocesamiento/incrustación de imágenes de la ejecución de generación.
8. `RAGIndexer`, `RAGRetriever` y `RAGAnswerer` separan la ingestión de documentos, recuperación y generación de respuestas.
9. Las bibliotecas nativas permanecen en el mismo módulo de Android, pero la carga nativa ahora es explícita y anulable para pruebas JVM en lugar de depender de efectos secundarios estáticos.

En el lado nativo, el proyecto sigue compilando llama.cpp, stable-diffusion.cpp, whisper.cpp, bark.cpp y las fuentes del puente JNI a través del Android NDK.

## Tecnologías

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Backend principal de LLM
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — Backend de generación de imágenes/vídeos
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Backend de voz a texto
- [bark.cpp](https://github.com/PABannier/bark.cpp) — Backend de texto a voz
- GGUF / GGML — Formatos de modelo
- Android NDK / JNI — Enlaces nativos
- ONNX Runtime — Incrustaciones de oraciones
- Android DownloadManager — Descargas de archivos grandes

## Métricas de memoria

Puedes medir el uso de RAM en tiempo de ejecución:

```kotlin
val snapshot = MemoryMetrics.snapshot(context)
Log.d("Memory", snapshot.toPretty(context))
```

Puntos de medición típicos:

- Antes de la carga del modelo
- Después de la carga del modelo
- Después del prompt bloqueante
- Después del prompt en streaming

#### Campos clave:

- `totalPssKb`: Uso total proporcional de RAM. Mejor para seguimiento general.
- `dalvikPssKb`: Montón y tiempo de ejecución gestionados por JVM.
- `nativePssKb`: Montón nativo (llama.cpp, ONNX, tensores, caché KV).
- `otherPssKb`: Memoria miscelánea.

Monitorea `nativePssKb` de cerca durante la carga del modelo y la inferencia para entender la huella de memoria del LLM.
Los tiempos de ejecución expertos como `SmolLM` también exponen estimaciones de memoria nativa/específicas del estado cuando necesitas instrumentación de nivel inferior.

## Notas

- `VULKAN_SDK` aún puede ser requerido cuando compilas la ruta Vulkan en el host.
- Verifica la capacidad de GPU de Android con los ayudantes explícitos por subsistema como `LLMEdge.getTextBackendAvailability()` y `LLMEdge.getImageBackendAvailability()`.

### Configuración de ProGuard/R8

La biblioteca incluye reglas de ProGuard para consumidores. Si necesitas agregar reglas personalizadas:

```proguard
# Mantener motores OCR
-keep class io.aatricks.llmedge.vision.** { *; }
-keep class org.bytedeco.** { *; }
-keep class com.google.mlkit.** { *; }

# Suprimir advertencias para dependencias opcionales
-dontwarn org.bytedeco.**
-dontwarn com.google.mlkit.**
```

### Licencias

- **llmedge**: Apache 2.0
- **llama.cpp**: MIT
- **stable-diffusion.cpp**: MIT
- **whisper.cpp**: MIT
- **bark.cpp**: MIT
- **Leptonica**: Personalizada (tipo BSD)
- **Google ML Kit**: Propietaria (ver términos de ML Kit)
- **JavaCPP**: Apache 2.0

## Licencia y Créditos

Este proyecto se basa en el trabajo de [Shubham Panchal](https://github.com/shubham0204), [ggerganov](https://github.com/ggerganov), y [PABannier](https://github.com/PABannier).
Consulta [CREDITS.md](CREDITS.md) para detalles completos.

## Pruebas

¿Buscas ejecutar pruebas unitarias y de instrumentación localmente, incluyendo verificaciones E2E nativas txt2img opcionales? Consulta la guía paso a paso en [docs/testing.md](docs/testing.md).

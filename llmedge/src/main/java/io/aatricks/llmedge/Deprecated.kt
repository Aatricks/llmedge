@file:Suppress("unused")
package io.aatricks.llmedge

import io.aatricks.llmedge.text.runtime.SmolLM as SmolLMImpl
import io.aatricks.llmedge.image.diffusion.StableDiffusion as StableDiffusionImpl
import io.aatricks.llmedge.speech.stt.Whisper as WhisperImpl
import io.aatricks.llmedge.speech.tts.BarkTTS as BarkTTSImpl
import io.aatricks.llmedge.runtime.ModelCache as ModelCacheImpl
import io.aatricks.llmedge.runtime.CpuTopology as CpuTopologyImpl
import io.aatricks.llmedge.runtime.FlashAttentionHelper as FlashAttentionHelperImpl
import io.aatricks.llmedge.runtime.GGUFReader as GGUFReaderImpl

@Deprecated("Moved to io.aatricks.llmedge.text.runtime.SmolLM", ReplaceWith("SmolLMImpl"))
typealias SmolLM = SmolLMImpl

@Deprecated("Moved to io.aatricks.llmedge.image.diffusion.StableDiffusion", ReplaceWith("StableDiffusionImpl"))
typealias StableDiffusion = StableDiffusionImpl

@Deprecated("Moved to io.aatricks.llmedge.speech.stt.Whisper", ReplaceWith("WhisperImpl"))
typealias Whisper = WhisperImpl

@Deprecated("Moved to io.aatricks.llmedge.speech.tts.BarkTTS", ReplaceWith("BarkTTSImpl"))
typealias BarkTTS = BarkTTSImpl

@Deprecated("Moved to io.aatricks.llmedge.runtime.ModelCache", ReplaceWith("ModelCacheImpl"))
typealias ModelCache<T> = ModelCacheImpl<T>

@Deprecated("Moved to io.aatricks.llmedge.runtime.CpuTopology", ReplaceWith("CpuTopologyImpl"))
typealias CpuTopology = CpuTopologyImpl

@Deprecated("Moved to io.aatricks.llmedge.runtime.FlashAttentionHelper", ReplaceWith("FlashAttentionHelperImpl"))
typealias FlashAttentionHelper = FlashAttentionHelperImpl

@Deprecated("Moved to io.aatricks.llmedge.runtime.GGUFReader", ReplaceWith("GGUFReaderImpl"))
typealias GGUFReader = GGUFReaderImpl

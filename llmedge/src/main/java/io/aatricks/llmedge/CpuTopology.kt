package io.aatricks.llmedge

import android.util.Log
import java.io.File

/**
 * Detects CPU topology to optimize thread allocation for big.LITTLE architectures. Modern ARM SoCs
 * have heterogeneous cores with different performance characteristics.
 */
object CpuTopology {
    private const val TAG = "CpuTopology"
    private const val CPU_BASE_PATH = "/sys/devices/system/cpu"
    private val CPU_DIR_REGEX = Regex("cpu\\d+")

    /** Information about CPU core configuration */
    data class CoreInfo(
            val totalCores: Int,
            val performanceCores: Int,
            val efficiencyCores: Int,
            val maxFrequencies: List<Long>
    ) {
        override fun toString(): String {
            return "Cores: $totalCores total ($performanceCores P-cores, $efficiencyCores E-cores)"
        }
    }

    /** Task types for thread count optimization */
    enum class TaskType {
        PROMPT_PROCESSING, // Latency-sensitive, use all P-cores
        TOKEN_GENERATION, // Throughput-oriented, use fewer cores
        DIFFUSION, // Compute-heavy, use maximum parallelism
        LIGHT_TASK // Quick operations, use 1-2 cores
    }

    private var cachedCoreInfo: CoreInfo? = null

    /**
     * Detects the CPU topology by reading core frequencies Higher max frequency cores are
     * classified as performance cores
     */
    fun detectCoreTopology(): CoreInfo {
        cachedCoreInfo?.let {
            return it
        }

        val maxFrequencies = mutableListOf<Long>()
        val cpuDir = File(CPU_BASE_PATH)

        if (!cpuDir.exists()) {
            Log.w(TAG, "CPU sysfs not available, using defaults")
            return createDefaultCoreInfo()
        }

        // Read max frequencies for each core
        val cpuDirs =
                cpuDir
                        .listFiles { file ->
                            file.isDirectory && file.name.matches(CPU_DIR_REGEX)
                        }
                        ?.sortedBy { it.name }
                        ?: emptyList()

        for (cpuDir in cpuDirs) {
            val freqFile = File(cpuDir, "cpufreq/cpuinfo_max_freq")
            if (freqFile.exists()) {
                try {
                    val freq = freqFile.readText().trim().toLongOrNull() ?: 0L
                    maxFrequencies.add(freq)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read frequency for ${cpuDir.name}: ${e.message}")
                    maxFrequencies.add(0L)
                }
            } else {
                // Core exists but no frequency info
                maxFrequencies.add(0L)
            }
        }

        if (maxFrequencies.isEmpty()) {
            return createDefaultCoreInfo()
        }

        // Classify cores: if there's a significant frequency gap, split into P/E cores
        val sortedFreqs = maxFrequencies.filter { it > 0 }.sorted()
        if (sortedFreqs.isEmpty()) {
            return createDefaultCoreInfo()
        }

        val maxFreq = sortedFreqs.last()
        val minFreq = sortedFreqs.first()
        val freqGap = maxFreq - minFreq

        // If frequency gap is >30%, we have big.LITTLE
        val hasBigLittle = freqGap > (maxFreq * 0.3)

        val performanceCores: Int
        val efficiencyCores: Int

        if (hasBigLittle) {
            // Threshold: 85% of max frequency
            val threshold = maxFreq * 0.85
            performanceCores = maxFrequencies.count { it >= threshold }
            efficiencyCores = maxFrequencies.count { it > 0 && it < threshold }
        } else {
            // Homogeneous cores (all same speed)
            performanceCores = maxFrequencies.size
            efficiencyCores = 0
        }

        val coreInfo =
                CoreInfo(
                        totalCores = maxFrequencies.size,
                        performanceCores = performanceCores,
                        efficiencyCores = efficiencyCores,
                        maxFrequencies = maxFrequencies
                )

        Log.i(TAG, "Detected CPU topology: $coreInfo")
        cachedCoreInfo = coreInfo
        return coreInfo
    }

    /** Get optimal thread count for a specific task type */
    fun getOptimalThreadCount(taskType: TaskType): Int {
        val coreInfo = detectCoreTopology()

        return when (taskType) {
            TaskType.PROMPT_PROCESSING -> {
                // Restore prior fast default: 4 threads for small LLMs (matches previous SmolLM default)
                // On larger devices, allow up to 16 threads but never below 4.
                val avail = Runtime.getRuntime().availableProcessors().coerceAtMost(16)
                avail.coerceAtLeast(4)
            }
            TaskType.TOKEN_GENERATION -> {
                // Use fewer cores to reduce contention
                // Token generation is sequential, so fewer threads can be faster
                if (coreInfo.performanceCores >= 4) {
                    (coreInfo.performanceCores / 2).coerceAtLeast(2)
                } else {
                    Runtime.getRuntime().availableProcessors().coerceAtMost(4)
                }
            }
            TaskType.DIFFUSION -> {
                // Revert to prior behavior: use all available CPU cores for maximum throughput.
                // This restores the faster generation speed observed before the facade refactor.
                Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
            }
            TaskType.LIGHT_TASK -> {
                // Use 1-2 cores for quick operations
                2
            }
        }
    }

    /** Creates default core info when detection fails */
    private fun createDefaultCoreInfo(): CoreInfo {
        val totalCores = Runtime.getRuntime().availableProcessors()
        return CoreInfo(
                totalCores = totalCores,
                performanceCores = totalCores,
                efficiencyCores = 0,
                maxFrequencies = emptyList()
        )
    }

    /** Returns true if device has big.LITTLE architecture */
    fun hasBigLittleArchitecture(): Boolean {
        val coreInfo = detectCoreTopology()
        return coreInfo.efficiencyCores > 0
    }

    /** Returns a bitmask of performance core indices for thread affinity pinning. */
    fun getPerformanceCoreMask(): Long {
        val coreInfo = detectCoreTopology()
        if (coreInfo.efficiencyCores == 0) return 0L // homogeneous, no pinning needed

        var mask = 0L
        val maxFreq = coreInfo.maxFrequencies.maxOrNull() ?: return 0L
        val threshold = maxFreq * 0.85

        for (i in coreInfo.maxFrequencies.indices) {
            if (coreInfo.maxFrequencies[i] >= threshold) {
                mask = mask or (1L shl i)
            }
        }
        return mask
    }
}

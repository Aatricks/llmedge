package io.aatricks.llmedge.image.ipc

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.Process
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.WorkerBindException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Owns the Binder connection to [DiffusionWorkerService]: bind on demand, death notification,
 * lazy reconnect (every (re)connect re-sends [WorkerInitConfig]), and the kill escalation the
 * watchdog and cancellation paths rely on.
 */
internal class WorkerConnectionManager(private val context: Context) {
    internal class Connection(
        val worker: IDiffusionWorker,
        val binder: IBinder,
        val pid: Int,
    ) {
        @Volatile var dead: Boolean = false

        /** Death listener for the request currently in flight on this connection. */
        @Volatile var onDeath: (() -> Unit)? = null
    }

    private val mutex = Mutex()
    private var current: Connection? = null
    private var serviceConnection: ServiceConnection? = null

    /** Binds (or reuses the live connection) and (re)initializes the worker with [initConfig]. */
    suspend fun connect(initConfig: WorkerInitConfig?): Connection =
        mutex.withLock {
            current?.takeIf { !it.dead && it.binder.isBinderAlive }?.let { live ->
                if (initConfig != null) {
                    live.worker.initialize(initConfig)
                }
                return live
            }
            unbindLocked()

            val binder = bindAndAwait()
            val worker = IDiffusionWorker.Stub.asInterface(binder)
            val connection = Connection(worker = worker, binder = binder, pid = worker.pid)
            binder.linkToDeath(
                {
                    connection.dead = true
                    AndroidLogAdapter.w(LOG_TAG, "Diffusion worker process died (pid=${connection.pid})")
                    connection.onDeath?.invoke()
                },
                0,
            )
            if (initConfig != null) {
                worker.initialize(initConfig)
            }
            current = connection
            return connection
        }

    private suspend fun bindAndAwait(): IBinder =
        suspendCancellableCoroutine { continuation ->
            val intent = Intent(context, DiffusionWorkerService::class.java)
            val serviceConn =
                object : ServiceConnection {
                    override fun onServiceConnected(
                        name: ComponentName?,
                        service: IBinder?,
                    ) {
                        if (continuation.isActive) {
                            if (service != null) {
                                continuation.resume(service)
                            } else {
                                continuation.resumeWithException(WorkerBindException("null binder"))
                            }
                        }
                    }

                    override fun onServiceDisconnected(name: ComponentName?) {
                        // Death is handled via linkToDeath on the binder.
                    }

                    override fun onBindingDied(name: ComponentName?) {
                        if (continuation.isActive) {
                            continuation.resumeWithException(WorkerBindException("binding died"))
                        }
                    }

                    override fun onNullBinding(name: ComponentName?) {
                        if (continuation.isActive) {
                            continuation.resumeWithException(WorkerBindException("null binding"))
                        }
                    }
                }
            serviceConnection = serviceConn
            val bound = context.bindService(intent, serviceConn, Context.BIND_AUTO_CREATE)
            if (!bound) {
                serviceConnection = null
                runCatching { context.unbindService(serviceConn) }
                if (continuation.isActive) {
                    continuation.resumeWithException(
                        WorkerBindException("bindService returned false; is the service declared in the merged manifest?"),
                    )
                }
            }
            continuation.invokeOnCancellation { runCatching { context.unbindService(serviceConn) } }
        }

    /** SIGKILL the worker. Same UID, so this is permitted; linkToDeath delivers the notification. */
    fun killWorker(connection: Connection) {
        AndroidLogAdapter.w(LOG_TAG, "Killing diffusion worker pid=${connection.pid}")
        Process.killProcess(connection.pid)
    }

    suspend fun close() {
        mutex.withLock { unbindLocked() }
    }

    private fun unbindLocked() {
        serviceConnection?.let { runCatching { context.unbindService(it) } }
        serviceConnection = null
        current = null
    }

    companion object {
        private const val LOG_TAG = "WorkerConnection"
    }
}

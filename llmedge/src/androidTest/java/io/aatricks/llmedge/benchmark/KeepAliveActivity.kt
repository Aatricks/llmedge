package io.aatricks.llmedge.benchmark

import android.app.Activity
import android.os.Bundle
import android.view.WindowManager

/**
 * Blank activity that keeps the instrumented process in the TOP oom-adj bucket while a
 * long-running native benchmark executes. Without a visible activity the test process is
 * a prime LMK candidate and multi-GB model loads (FLUX.2) get SIGKILLed under pressure.
 */
class KeepAliveActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
}

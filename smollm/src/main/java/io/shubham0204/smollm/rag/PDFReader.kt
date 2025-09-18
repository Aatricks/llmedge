package io.shubham0204.smollm.rag

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper

/**
 * Reads a PDF file (from a Uri) and returns its extracted text.
 */
object PDFReader {
    suspend fun readAllText(context: Context, uri: Uri): String = withContext(Dispatchers.IO) {
        // Init PDFBox once
        try { PDFBoxResourceLoader.init(context) } catch (_: Throwable) {}
        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open PDF Uri: $uri" }
            PDDocument.load(input).use { doc ->
                val stripper = PDFTextStripper()
                stripper.sortByPosition = true
                stripper.startPage = 1
                stripper.endPage = doc.numberOfPages
                return@use stripper.getText(doc)
            }
        }
    }
}

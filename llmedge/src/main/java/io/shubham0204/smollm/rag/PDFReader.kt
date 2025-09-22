/*
 * Copyright (C) 2025 Aatricks
 *
 * Licensed under the GNU General Public License v3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.gnu.org/licenses/gpl-3.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

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

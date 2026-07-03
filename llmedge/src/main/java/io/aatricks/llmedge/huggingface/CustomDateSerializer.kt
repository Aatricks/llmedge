/*
 * Copyright (C) 2025 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.aatricks.llmedge.huggingface

import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/**
 * A custom serializer implementation for the java.time.LocalDateTime class.
 */
class CustomDateSerializer : KSerializer<LocalDateTime> {
    override val descriptor: SerialDescriptor =
        PrimitiveSerialDescriptor("LocalDateTime", PrimitiveKind.STRING)

    private val utcFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")

    override fun serialize(
        encoder: Encoder,
        value: LocalDateTime,
    ) {
        encoder.encodeString(value.atOffset(ZoneOffset.UTC).format(utcFormatter))
    }

    override fun deserialize(decoder: Decoder): LocalDateTime {
        val stringRepr = decoder.decodeString()
        return try {
            try {
                val temporalAccessor = DateTimeFormatter.ISO_INSTANT.parse(stringRepr)
                val instant = java.time.Instant.from(temporalAccessor)
                LocalDateTime.ofInstant(instant, ZoneOffset.UTC)
            } catch (_: Throwable) {
                try {
                    val temporalAccessor = DateTimeFormatter.ISO_OFFSET_DATE_TIME.parse(stringRepr)
                    val instant = java.time.Instant.from(temporalAccessor)
                    LocalDateTime.ofInstant(instant, ZoneOffset.UTC)
                } catch (_: Throwable) {
                    try {
                        LocalDateTime.parse(stringRepr)
                    } catch (_: Throwable) {
                        LocalDateTime.parse(stringRepr, utcFormatter)
                    }
                }
            }
        } catch (e: Throwable) {
            LocalDateTime.ofEpochSecond(0L, 0, ZoneOffset.UTC)
        }
    }
}

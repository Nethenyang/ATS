# Data sources, provenance and attribution

This bundle contains source manifests and numerical measurements. It contains no original, prepared or encoded audio, and assigns no new rights over third-party recordings.

## LibriSpeech dev-clean

Official source: [OpenSLR SLR12](https://www.openslr.org/12). Acquisition URL used by the code: `https://www.openslr.org/resources/12/dev-clean.tar.gz`.

OpenSLR identifies LibriSpeech as **CC BY 4.0**. Attribute the corpus creators and consult the official release terms when acquiring or redistributing their material. The evaluated subset uses one six-second excerpt per dev-clean speaker, selected by the rule in the README. Public manifests identify the original archive members; prepared-reference SHA-256 values apply to the float WAVs produced by preprocessing, not to the original FLAC members.

Reference:

> Panayotov, V., Chen, G., Povey, D., Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. ICASSP, 5206–5210. DOI: [10.1109/ICASSP.2015.7178964](https://doi.org/10.1109/ICASSP.2015.7178964).

## MUSDB18 official seven-second sample release

Official dataset description: [SigSep MUSDB18](https://sigsep.github.io/datasets/musdb.html). Acquisition URL: `https://github.com/sigsep/sigsep-mus-db/releases/download/v0.4.0/MUSDB18-7-STEMS.zip`.

MUSDB18 combines recordings with different source conditions; its official documentation lists the component sources and licenses. It is not described here as one universally unrestricted dataset. This repository does not redistribute its audio or apply the LibriSpeech license to music recordings.

The evaluated data are all 50 test tracks in the published short sample archive, mixture stream 0, downmixed by the arithmetic stereo mean. Although the release is nominally seven seconds, all 50 prepared music references in the recorded run are 6.8 seconds long. Preprocessing keeps available decoded samples and does not pad them to seven seconds. The source streams are already AAC-compressed. Quality scores therefore describe further transcoding relative to those decoded samples, not comparison to uncompressed studio masters. The recorded `excerpt_start_seconds = 0` refers to the beginning of the downloaded short sample, not to an asserted position in the original full song; the original-song offsets are not recovered in this bundle.

Reference:

> Rafii, Z., Liutkus, A., Stöter, F.-R., Mimilakis, S. I., Bittner, R. (2017). The MUSDB18 corpus for music separation. Zenodo. DOI: [10.5281/zenodo.1117372](https://doi.org/10.5281/zenodo.1117372).

## Supplied competition recordings

The supplied materials came from the authors' 2025 MathorCup problem-C files. Two clean references are used in the main encoding experiment. Existing coded variants are audited separately. Two noisy stereo clips are used only in auxiliary reference-free diagnostics.

The bundle includes identifiers, hashes and measurements but no competition audio or problem statement. No public acquisition URL is asserted. Reproducing these cases requires separately obtained originals; public-only reproduction remains possible without them.

## Metadata files

- `public_sources.json`: the 90 originally evaluated public source records, using portable relative paths.
- `librispeech_manifest.csv`: 40 speakers and selected archive utterances.
- `musdb18_manifest.csv`: 50 published test samples and mixture identifiers.
- `supplied_audio_manifest.csv`: the two supplied main references and their expected hashes.
- `download_provenance.json`: original archive byte lengths and SHA-256 hashes.
- `results/recorded/source_manifest.json`: the complete 92-source mapping used by the recorded measurements.

Different sources retain their own attribution and usage conditions. Repository-level software/result licensing has not been selected in this preparation bundle.

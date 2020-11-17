
1. Create folder with model name. E.g. `my-little-model`
2. Create `.yml` with model name. E.g. `model.yml`
3. Use a `.yml` template e.g. `mobilenet-ssd/mobilenet-ssd.yml`
Change the fields in `.yml` accordingly to the model.
`size` can be obtained: `wc -c < filename`
`sha256` can be obtained: `sha256sum filename`
`name` : "FP16/`model_name`.xml/bin" e.g. `FP16/my-little-model.xml`
`source:`
* Model stored on google drive (public link):
```
      $type: google_drive
      id: 11-PX4EDxAnhymbuvnyb91ptvZAW3oPOn
```
Where `id` is the `id` of the link. E.g. in https://drive.google.com/file/d/1pdC4eNWxyfewCJ7T0i9SXLHKt39gBDZV/view?usp=sharing 
`id` is `1pdC4eNWxyfewCJ7T0i9SXLHKt39gBDZV`

* Model not stored on google drive: `source` is the link to the file to download.

`framework:` `dldt` only for now.
`license:` license file to the model.

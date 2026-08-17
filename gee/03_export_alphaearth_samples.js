// AlphaEarth 2022 point extraction fallback for the Earth Engine Code Editor.
// Upload data_metadata/alphaearth_sample_manifest.csv as a table, then replace
// the placeholder asset ID below. The manifest already preserves region,
// sample_id, label source and frozen split membership.
var samples = ee.FeatureCollection('projects/REPLACE_ME/assets/alphaearth_sample_manifest');
var bands = ee.List.sequence(0, 63).map(function(i) {
  return ee.String('A').cat(ee.Number(i).format('%02d'));
});
var embedding = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate('2022-01-01', '2023-01-01')
  .mosaic()
  .select(bands);
var sampled = embedding.sampleRegions({
  collection: samples,
  properties: ['sample_id', 'region', 'year', 'class_id', 'label_source', 'spatial_block', 'split'],
  scale: 10,
  geometries: false,
  tileScale: 4
});
print('input samples', samples.size());
print('exported samples', sampled.size());
print('first row', sampled.first());
Export.table.toDrive({
  collection: sampled,
  description: 'alphaearth_2022_jiangxi_shanghai_samples',
  fileNamePrefix: 'alphaearth_2022_jiangxi_shanghai_samples',
  fileFormat: 'CSV',
  selectors: ee.List(['sample_id', 'region', 'year', 'class_id', 'label_source', 'spatial_block', 'split']).cat(bands)
});

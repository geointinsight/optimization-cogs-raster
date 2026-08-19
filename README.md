<div align="center">

# COG Optimization Benchmark

**Measuring efficient cloud-native raster access with Cloud Optimized GeoTIFFs**

![COG](https://img.shields.io/badge/Format-COG-2f855a?style=flat-square)
![Cloud Native](https://img.shields.io/badge/Workflow-Cloud--Native-2563eb?style=flat-square)
![Python](https://img.shields.io/badge/Language-Python-f59e0b?style=flat-square&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-S3--compatible-7c3aed?style=flat-square)

</div>

---

## Objective

Evaluate how COG compression methods and tile sizes affect cloud-based raster access.

The benchmark focuses on:

| Focus area | Measurement |
|---|---|
| **Creation** | COG creation time |
| **Storage** | Output file size |
| **Spatial access** | Tiles touched by a BBOX request |
| **Data transfer** | Compressed bytes read through HTTP Range requests |
| **User experience** | Cold and warm BBOX read latency on S3-compatible storage |

> **Goal** — identify a practical COG configuration for interactive geospatial applications that read partial raster windows from object storage.

## Tools Used

<table>
  <tr>
    <th>Category</th>
    <th>Tools</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td><strong>COG processing</strong></td>
    <td>GDAL, Rasterio</td>
    <td>Create, validate, inspect, and read COG windows</td>
  </tr>
  <tr>
    <td><strong>Data processing</strong></td>
    <td>Python, Fiona, NumPy</td>
    <td>Benchmark orchestration, vector BBOX input, and raster-array handling</td>
  </tr>
  <tr>
    <td><strong>Cloud access</strong></td>
    <td>Boto3, HTTP Range requests</td>
    <td>Access S3-compatible storage and verify partial tile reads</td>
  </tr>
  <tr>
    <td><strong>Visualization</strong></td>
    <td>Matplotlib, HTML, CSS, JavaScript</td>
    <td>Generate charts and present interactive benchmark results</td>
  </tr>
  <tr>
    <td><strong>Data format</strong></td>
    <td>CSV</td>
    <td>Store reproducible benchmark measurements</td>
  </tr>
</table>

<div align="center">

<sub>Cloud-native raster benchmarking for measurable storage and read performance.</sub>

</div>

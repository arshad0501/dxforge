# DxForge Benchmark Report

- Cases: **20**
- Request structural validity: **100.0%**
- Request byte correctness: **100.0%**
- Outcome (positive/negative) correctness: **100.0%**
- NRC correctness (negative cases): **100.0%**
- Retrieval recall@4 (relevant evidence): **95.0%**
- Latency p50 / p95: **0.03 ms / 0.05 ms**

| Case | Intent | Expected | Actual | req | out | nrc | recall |
|------|--------|----------|--------|-----|-----|-----|--------|
| BM-01 | read the VIN | 22 F1 90 / positive | 22 F1 90 / positive | ok | ok | ok | ok |
| BM-02 | read the ECU serial number | 22 F1 8C / positive | 22 F1 8C / positive | ok | ok | ok | ok |
| BM-03 | read the software version | 22 F1 89 / positive | 22 F1 89 / positive | ok | ok | ok | ok |
| BM-04 | read data identifier 0x0105 | 22 01 05 / positive | 22 01 05 / positive | ok | ok | ok | ok |
| BM-05 | read DID 0x9999 | 22 99 99 / negative 0x31 | 22 99 99 / negative 0x31 | ok | ok | ok | X |
| BM-06 | write door lock state | 2E 41 01 / negative 0x7f | 2E 41 01 / negative 0x7f | ok | ok | ok | ok |
| BM-07 | write door lock state | 2E 41 01 / positive | 2E 41 01 / positive | ok | ok | ok | ok |
| BM-08 | change to extended diagnostic session | 10 03 / positive | 10 03 / positive | ok | ok | ok | ok |
| BM-09 | enter programming session | 10 02 / positive | 10 02 / positive | ok | ok | ok | ok |
| BM-10 | hard reset the ECU | 11 01 / positive | 11 01 / positive | ok | ok | ok | ok |
| BM-11 | read all fault codes | 19 02 / positive | 19 02 / positive | ok | ok | ok | ok |
| BM-12 | clear diagnostic information | 14 FF FF FF / positive | 14 FF FF FF / positive | ok | ok | ok | ok |
| BM-13 | send tester present | 3E 00 / positive | 3E 00 / positive | ok | ok | ok | ok |
| BM-14 | start the self test routine | 31 01 03 01 / positive | 31 01 03 01 / positive | ok | ok | ok | ok |
| BM-15 | erase memory routine | 31 01 02 03 / negative 0x33 | 31 01 02 03 / negative 0x33 | ok | ok | ok | ok |
| BM-16 | write calibration set | 2E 42 00 / negative 0x33 | 2E 42 00 / negative 0x33 | ok | ok | ok | ok |
| BM-17 | request security seed | 27 01 / positive | 27 01 / positive | ok | ok | ok | ok |
| BM-18 | disable communication | 28 03 01 / positive | 28 03 01 / positive | ok | ok | ok | ok |
| BM-19 | turn off dtc setting | 85 02 / positive | 85 02 / positive | ok | ok | ok | ok |
| BM-20 | read the active diagnostic session | 22 F1 86 / positive | 22 F1 86 / positive | ok | ok | ok | ok |

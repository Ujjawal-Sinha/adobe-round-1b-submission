# PDF Semantic Analyzer


## Prerequisites

-   Python 3.x
-   Docker (for containerized usage)

## Local Usage

*1. Setup*

Place your PDF files in an input/ directory.:

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```
*2. Run*
```bash
docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output --network none mysolutionname:somerandomidentifier
```

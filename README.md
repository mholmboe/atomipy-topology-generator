# atomipy topology generator

The **atomipy topology generator** is a powerful web-based application designed to automate the complex process of atom typing and molecular topology generation for mineral and material systems. 

Powered by the [**atomipy**](https://github.com/mholmboe/atomipy) core library, this tool provides a streamlined interface for preparing structures for molecular dynamics (MD) simulations using the **MINFF** and **CLAYFF** forcefields.

---

## 🚀 Key Features

- **Automated Atom Typing**: Intelligent assignment of atom types and partial charges based on local coordination environments for the MINFF and CLAYFF forcefields.
- **Multi-Format Support**: Import structures from **PDB**, **GRO**, **XYZ**, and **CIF/mmCIF** files.
- **Topology Export**: Generate ready-to-use topology files for:
  - **GROMACS** (`.itp`)
  - **LAMMPS** (`.data`)
  - **NAMD / OpenMM** (`.psf`)
- **Performance Optimized**: Specifically engineered to handle large systems (tested up to 50,000+ atoms), with typical processing times between 10-15 minutes for complex minerals.
- **Crystallographic Awareness**: Full support for both orthogonal and triclinic simulation cells with periodic boundary conditions (PBC).

---

## 🛠️ Technology Stack

- **Backend**: Flask (Python 3.11+)
- **Analysis Engine**: `atomipy` core package
- **Frontend**: Modern HTML5/CSS with a responsive design
- **Deployment**: Optimized for **Google Cloud Run** and **Render** via Docker.

---

## 💻 Local Setup

### Running with Docker (Recommended)

The easiest way to run the generator locally is using the provided `Dockerfile`:

```bash
# Build the image
docker build -t atomipy-topology-generator .

# Run the container
docker run -p 5001:5001 -e PORT=5001 atomipy-topology-generator
```

Access the app at `http://localhost:5001`.

### Running with Python

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python app.py
   ```

---

## 📂 Project Structure

- `app.py`: The main Flask application and API logic.
- `atomipy/`: The core library providing the analysis and topology generation engine.
- `templates/`: HTML structures and page layouts.
- `static/`: CSS styling, brand assets, and client-side scripts.
- `uploads/` / `results/`: Temporary storage for user-uploaded structures and generated bundles.

---

## 📄 License & Credits

Developed by **M. Holmboe**. 

This application is built upon the research and development of the `atomipy` Python toolbox, which was inspired by the original MATLAB [**atom**](https://github.com/mholmboe/atom) library.

For the core engine source and advanced Python API usage, visit the [**atomipy repository**](https://github.com/mholmboe/atomipy).

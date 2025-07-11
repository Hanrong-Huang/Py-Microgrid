## **Project Change Request: Py-Microgrid Enhancements**

This document outlines a series of requested changes and improvements for the `py_microgrid` repository. The goal is to enhance modularity, user-friendliness, and analytical flexibility.

### **1. Refactor `grid` to `genset` and Introduce a True `grid` Component**

The current system architecture uses a module named `grid` to simulate a backup generator (genset). This request restructures the components to be more intuitive and functionally accurate.

**Actionable Steps:**

*   **Create `genset` Module:**
    *   Rename the existing `grid` module to `genset`.
    *   This new `genset` module will retain the original logic: it acts as a backup power source, activating when renewable energy supply is insufficient to meet the load. It represents a dispatchable, on-site generator.
*   **Create New `grid` Module:**
    *   Introduce a new, separate `grid` module that represents a connection to the main electrical grid.
    *   This module's behavior should be governed by the `grid_resource_file`: `py_microgrid/simulation/resource_files/grid/dispatch_factors_ts.csv`. This file should determine the grid's availability, price, or dispatch schedule.
    *   **Functionality:**
        *   When the `grid` component is **enabled** (`true` in YAML), the microgrid operates in **grid-connected mode**. It can import power from the grid when local generation is insufficient or export excess power to the grid (if supported by the model).
        *   When the `grid` component is **disabled** (`false` in YAML), the microgrid operates in **off-grid (or islanded) mode**. It must rely solely on its other enabled assets (`solar`, `wind`, `genset`, `battery`).
*   **Update Configuration:**
    *   The main YAML configuration file must be updated to list all five components: `solar`, `wind`, `genset`, `battery`, and `grid`.
    *   Each component must have a boolean switch (e.g., `enabled: true/false`) to allow users to easily toggle them.

### **2. Update Optimization and Scripts for New Component Structure**

The introduction of a distinct `grid` component requires updates to the optimization algorithm and supporting scripts to ensure simulation integrity and backwards compatibility.

**Actionable Steps:**

*   **Update Nelder-Mead Optimization:**
    *   Modify the optimization function to include decision variables associated with the new `grid` component (e.g., size of grid connection, rules for import/export).
    *   Ensure the algorithm correctly incorporates the five distinct components into its cost-minimization or other objective functions.
*   **Validate Backwards Compatibility:**
    *   A critical validation test is required. Run a simulation with the following configuration in the **updated** software:
        *   `solar: enabled`
        *   `wind: enabled`
        *   `genset: enabled` (using the new `genset` module)
        *   `battery: enabled`
        *   `grid: disabled`
    *   The final optimization results (e.g., component sizes, total cost) from this test **must be identical** to the results produced by the **old version** of the software running with its original four components (`solar`, `wind`, `grid` (as genset), `battery`). This proves that the core logic has been preserved.
*   **Ensure Script Consistency:**
    *   Review and update all relevant scripts (e.g., data processors, result analyzers, plotting tools) to correctly handle the new five-component architecture.

### **3. Analyze and Potentially Remove Site Parameters from YAML**

The `site` section in the YAML configuration contains several parameters whose utility is unclear. This task is to determine their necessity.

**Actionable Steps:**

*   **Code Audit:**
    *   Perform a thorough search across the entire codebase to identify any usage of the following `site` parameters: `elev`, `lat`, `lon`, `site_boundaries`, `tz`, `urdb_label`, `year`.
*   **Decision and Rationale:**
    *   **Keep if Used:** If any of these parameters are actively used in calculations (e.g., `lat`, `lon`, `elev` are often essential for solar radiation and wind resource models), they must be kept. The reason for keeping them should be documented (e.g., "Latitude and longitude are required by the solar model to calculate sun position.").
    *   **Remove if Unused:** If the parameters are not used in any calculations, they are legacy artifacts and should be removed from the example YAML files to simplify configuration and reduce user confusion.
*   **Reflect Changes:**
    *   If any parameters are removed, ensure that the scripts responsible for parsing the YAML file are updated to no longer require them, preventing errors.

### **4. Simplify `desired_schedule` Input in YAML**

The current practice of embedding the entire 8760-hour `desired_schedule` directly into the YAML file makes it unwieldy.

**Actionable Steps:**

*   **Implement File Path Loading:**
    *   Modify the YAML parsing logic to accept a file path for the `desired_schedule`.
    *   The user should be able to specify the load profile like this:
        ```yaml
        # In input_file_chunk_0.yaml
        desired_schedule: 'load_data/Load data.csv'
        ```
*   **Update Data Loading Logic:**
    *   The script responsible for the `battery_dispatch: heuristic_load_following` logic must be updated. It should now check if the `desired_schedule` value is a list (for backward compatibility) or a string. If it's a string, it should treat it as a relative path and load the time-series data from the specified CSV file.

### **5. Externalize Configuration of Efficiency and Cost**

Key performance and cost parameters are currently hardcoded, limiting flexibility. This task is to make them user-configurable.

**Best Practice Recommendation:**

It is **strongly recommended** to allow users to modify these parameters at a top level (e.g., in the main YAML input file) rather than keeping them hardcoded.

*   **Why this is best practice:**
    *   **Flexibility & Research:** Users can easily run sensitivity analyses, compare different technology costs, or model future scenarios without ever touching the source code. This is critical for research and practical application.
    *   **User-Friendliness:** It empowers users who are not developers to fully utilize the tool. They should not have to navigate the codebase to change a simple cost assumption.
    *   **Maintainability:** It prevents accidental introduction of bugs into the source code when users attempt to modify hardcoded values. The core logic remains untouched.
    *   **Clarity:** The YAML file becomes a self-contained definition of the entire simulation scenario, making results easier to reproduce and understand.

**Actionable Steps:**

*   **Add Parameters to YAML:**
    *   Introduce new, clearly defined sections in the YAML file for technology-specific parameters. For example:
        ```yaml
        # In input_file_chunk_0.yaml
        technologies:
          solar:
            efficiency: 0.25  # User-definable PV efficiency
            # other solar params...
          wind:
            efficiency: 0.35  # User-definable wind turbine efficiency
            # other wind params...

        financials:
          costs:
            solar_per_kw: 900
            wind_per_kw: 1400
            genset_per_kw: 500
            battery_per_kwh: 300
            # ... other cost parameters
        ```
*   **Update Code to Use YAML Values:**
    *   Modify the `pv` and `wind` simulation modules to read their respective `efficiency` values from the parsed YAML configuration.
    *   Update the `cost_calculator.py` script to source all cost values from the `financials.costs` section of the YAML input.
*   **Update Examples:**
    *   Revise the `quick_start_example.ipynb` and other user-facing examples to showcase how to define and modify these new parameters in the YAML file.

### **6. Other Suggested Improvements**

To further increase the quality and usability of the repository, consider the following:

*   **Add Unit and Integration Tests:** Create a `tests/` directory. Add unit tests for core physics modules (e.g., solar output given certain inputs) and integration tests to verify that components work together as expected (e.g., battery charges from solar).
*   **Enhance Documentation:** Improve function docstrings to explain inputs, outputs, and the purpose of each function. A more detailed project `README.md` explaining the new architecture would be highly beneficial.
*   **Formalize Dependencies:** Create a `requirements.txt` (or `pyproject.toml`) file to lock down project dependencies. This ensures that all users are running on a consistent and tested set of libraries, improving reproducibility.
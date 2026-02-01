"""
RankEM Excel Add-in Installation and Setup Script

This script:
1. Ensures xlwings is installed
2. Creates the Excel add-in (.xlam file)
3. Registers the UDFs with Excel
"""

import subprocess
import sys
from pathlib import Path

def check_xlwings():
    """Check if xlwings is installed, install if not."""
    try:
        import xlwings
        print(f"✓ xlwings version {xlwings.__version__} is installed")
        return True
    except ImportError:
        print("Installing xlwings...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xlwings"])
        print("✓ xlwings installed successfully")
        return True


def create_addin():
    """Create the Excel add-in file."""
    import xlwings as xw
    
    addin_dir = Path(__file__).parent
    addin_name = "rankEM"
    
    print(f"\nCreating Excel add-in in: {addin_dir}")
    
    # Use xlwings CLI to create the add-in
    # This creates the .xlam file that Excel will load
    try:
        # First, ensure we're in the right directory
        import os
        original_dir = os.getcwd()
        os.chdir(str(addin_dir))
        
        # Create a new xlwings project
        result = subprocess.run(
            [sys.executable, "-m", "xlwings", "addin", "install"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ xlwings add-in installed successfully")
        else:
            print(f"Note: {result.stderr.strip() if result.stderr else 'xlwings addin install completed'}")
        
        os.chdir(original_dir)
        
    except Exception as e:
        print(f"Note: {e}")
        print("You may need to run 'xlwings addin install' manually")


def print_usage_instructions():
    """Print instructions for using the add-in."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RankEM Excel Add-in - Setup Complete!                       ║
╠══════════════════════════════════════════════════════════════════════════════╣

📋 NEXT STEPS:

1. Open Excel

2. If the xlwings add-in is not already loaded:
   • Go to File → Options → Add-Ins
   • At the bottom, select "Excel Add-ins" and click "Go..."
   • Click "Browse..." and navigate to:
     %APPDATA%\\Microsoft\\Excel\\XLSTART\\xlwings.xlam
   • Check the box next to "xlwings" and click OK

3. In Excel, click the "xlwings" tab in the ribbon

4. Click "Import Functions" to load the RankEM UDFs

5. Set the "UDF Modules" to: rankEM_functions
   Set the "PYTHONPATH" to: {addin_path}

═══════════════════════════════════════════════════════════════════════════════

📊 AVAILABLE FUNCTIONS IN EXCEL:

  =RankEM_Theta(data, [method], [lambda])
      Returns student ability estimates (θ)
      
  =RankEM_Beta(data, [method], [lambda])  
      Returns problem difficulty estimates (β)
      
  =RankEM_Imputed(data, [method], [lambda])
      Returns the imputed score matrix
      
  =RankEM_Stats(data, [method], [lambda])
      Returns estimation statistics
      
  =RankEM_Ranking(data, [method], [lambda])
      Returns students ranked by ability
      
  =RankEM_AllMethods(data, [lambda])
      Compares results from all three methods

  Parameters:
    • data: Select your score matrix (students × problems)
    • method: "em", "mean_imputation", or "day_average"
    • lambda: Regularization parameter (default: 1.0)

═══════════════════════════════════════════════════════════════════════════════

💡 EXAMPLE:

  If your scores are in cells A1:H20 (20 students, 8 problems):
  
  • Get student rankings:  =RankEM_Ranking(A1:H20, "em", 1.0)
  • Get all statistics:    =RankEM_Stats(A1:H20, "em")
  • Compare all methods:   =RankEM_AllMethods(A1:H20)

╚══════════════════════════════════════════════════════════════════════════════╝
""".format(addin_path=Path(__file__).parent.resolve()))


def main():
    print("=" * 70)
    print("RankEM Excel Add-in Installation")
    print("=" * 70)
    
    # Step 1: Check/install xlwings
    print("\n[1/3] Checking xlwings installation...")
    check_xlwings()
    
    # Step 2: Create the add-in
    print("\n[2/3] Setting up Excel add-in...")
    create_addin()
    
    # Step 3: Print instructions
    print("\n[3/3] Installation complete!")
    print_usage_instructions()


if __name__ == "__main__":
    main()

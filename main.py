# Importing the needed modules 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
from dotenv import load_dotenv
import json, os
from Utils.logger import setup_logger
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Loading API key from a dotenv file.
load_dotenv(dotenv_path='.env')

# Setup logger
logger = setup_logger(__name__)

def main():
    try:
        # Read the medical report
        medical_report_path = os.path.join("Medical Reports", "Medical Rerort - Michael Johnson - Panic Attack Disorder.txt")
        with open(medical_report_path, 'r', encoding='utf-8') as file:
            medical_report = file.read()
            logger.info("Successfully read medical report")

        # Initialize specialist agents
        logger.info("Initializing specialist agents...")
        cardiologist = Cardiologist(medical_report)
        psychologist = Psychologist(medical_report)
        pulmonologist = Pulmonologist(medical_report)

        # Get specialist reports
        logger.info("Getting specialist reports...")
        cardiologist_report = cardiologist.run(medical_report)
        psychologist_report = psychologist.run(medical_report)
        pulmonologist_report = pulmonologist.run(medical_report)

        # Initialize multidisciplinary team
        logger.info("Initializing multidisciplinary team...")
        team_agent = MultidisciplinaryTeam(
            cardiologist_report=cardiologist_report,
            psychologist_report=psychologist_report,
            pulmonologist_report=pulmonologist_report
        )

        # Get final diagnosis
        logger.info("Getting final diagnosis...")
        final_diagnosis = team_agent.run(team_agent.data)

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Write the final diagnosis to a file
        logger.info("Writing final diagnosis to file...")
        final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis
        with open(os.path.join("results", "final_diagnosis.txt"), 'w', encoding='utf-8') as file:
            file.write(final_diagnosis_text)
            logger.info("Successfully wrote final diagnosis to file")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()



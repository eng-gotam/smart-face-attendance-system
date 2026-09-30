import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from face_recognition import recognize_face


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Smart Face Attendance",
    page_icon="👤",
    layout="wide"
)


# --------------------------------
# Custom CSS
# --------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------
# Title
# --------------------------------

st.markdown(
    '<div class="main-title">👤 Smart Face Attendance System</div>',
    unsafe_allow_html=True
)

st.write(
    "Real-time face recognition and automatic attendance system."
)


# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.title("⚙️ Controls")

st.sidebar.info(
    "Recognition is powered by OpenCV YuNet + SFace."
)


# --------------------------------
# Attendance file
# --------------------------------

attendance_file = "attendance.csv"


try:

    attendance_df = pd.read_csv(
        attendance_file
    )

except:

    attendance_df = pd.DataFrame(
        columns=[
            "Name",
            "Date",
            "Time",
            "Status"
        ]
    )


# --------------------------------
# Today's date
# --------------------------------

today = datetime.now().strftime(
    "%Y-%m-%d"
)


today_attendance = attendance_df[
    attendance_df["Date"] == today
]


# --------------------------------
# Dashboard statistics
# --------------------------------

col1, col2, col3 = st.columns(3)


col1.metric(
    "👥 Total Attendance Today",
    len(today_attendance)
)


col2.metric(
    "📅 Today's Date",
    today
)


col3.metric(
    "🕐 Current Time",
    datetime.now().strftime("%H:%M:%S")
)


st.write("---")


# --------------------------------
# Main layout
# --------------------------------

camera_col, attendance_col = st.columns(
    [2, 1]
)


# --------------------------------
# Camera
# --------------------------------

with camera_col:

    st.subheader("📷 Camera")

    camera_image = st.camera_input(
        "Take a photo for face recognition"
    )


    if camera_image is not None:

        # --------------------------------
        # Convert uploaded camera image
        # to OpenCV format
        # --------------------------------

        image_bytes = camera_image.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            st.error(
                "Could not process camera image."
            )

        else:

            # --------------------------------
            # Face recognition
            # --------------------------------

            name, score = recognize_face(
                frame
            )


            # --------------------------------
            # Attendance
            # --------------------------------

            if name == "No Face":

                st.warning(
                    "⚠️ No face detected. "
                    "Please take another photo."
                )


            elif name == "Unknown":

                st.warning(
                    f"⚠️ Unknown person "
                    f"(similarity: {score:.2f})"
                )


            else:

                current_time = datetime.now().strftime(
                    "%H:%M:%S"
                )


                already_present = False


                if not attendance_df.empty:

                    already_present = (
                        (
                            attendance_df["Name"]
                            == name
                        )
                        &
                        (
                            attendance_df["Date"]
                            == today
                        )
                    ).any()


                if not already_present:

                    new_row = pd.DataFrame(
                        [{
                            "Name": name,
                            "Date": today,
                            "Time": current_time,
                            "Status": "Present"
                        }]
                    )


                    attendance_df = pd.concat(
                        [
                            attendance_df,
                            new_row
                        ],
                        ignore_index=True
                    )


                    attendance_df.to_csv(
                        attendance_file,
                        index=False
                    )


                    st.success(
                        f"✅ Attendance marked: "
                        f"{name} "
                        f"(similarity: {score:.2f})"
                    )


                else:

                    st.info(
                        f"👤 {name} is already "
                        f"marked present today."
                    )


            # --------------------------------
            # Display captured image
            # --------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            st.image(
                frame_rgb,
                caption="Captured Image",
                width="stretch"
            )


# --------------------------------
# Attendance table
# --------------------------------

with attendance_col:

    st.subheader("📋 Today's Attendance")


    if today_attendance.empty:

        st.info(
            "No attendance recorded today."
        )


    else:

        st.dataframe(
            today_attendance,
            width="stretch",
            hide_index=True
        )


    # --------------------------------
    # Download attendance
    # --------------------------------

    csv_data = attendance_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📥 Download Attendance CSV",
        data=csv_data,
        file_name="attendance.csv",
        mime="text/csv"
    )


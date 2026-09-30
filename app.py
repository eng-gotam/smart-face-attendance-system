
import streamlit as st
import cv2
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

    .status-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
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

start_camera = st.sidebar.checkbox(
    "Start Camera"
)

st.sidebar.write("---")

st.sidebar.info(
    "Recognition is powered by "
    "OpenCV YuNet + SFace."
)


# --------------------------------
# Dashboard statistics
# --------------------------------

col1, col2, col3 = st.columns(3)


# Load attendance
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


today = datetime.now().strftime(
    "%Y-%m-%d"
)


today_attendance = attendance_df[
    attendance_df["Date"] == today
]


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

    st.subheader("📷 Live Camera")

    camera_placeholder = st.empty()

    status_placeholder = st.empty()


    if start_camera:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            st.error(
                "Could not open webcam."
            )

        else:

            while start_camera:

                ret, frame = cap.read()

                if not ret:

                    st.error(
                        "Could not read webcam frame."
                    )

                    break


                # Recognize faces
                processed_frame, results = recognize_face(
                    frame
                )


                # --------------------------------
                # Attendance
                # --------------------------------

                for result in results:

                    name = result["name"]
                    score = result["score"]


                    if name != "Unknown":

                        today = datetime.now().strftime(
                            "%Y-%m-%d"
                        )

                        current_time = datetime.now().strftime(
                            "%H:%M:%S"
                        )


                        # Check duplicate attendance
                        already_present = False

                        if not attendance_df.empty:

                            already_present = (
                                (
                                    attendance_df["Name"] == name
                                )
                                &
                                (
                                    attendance_df["Date"] == today
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


                            status_placeholder.success(
                                f"✅ Attendance marked: {name}"
                            )

                        else:

                            status_placeholder.info(
                                f"👤 {name} already marked present today."
                            )

                    else:

                        status_placeholder.warning(
                            "⚠️ Unknown person detected."
                        )


                # Convert BGR → RGB
                frame_rgb = cv2.cvtColor(
                    processed_frame,
                    cv2.COLOR_BGR2RGB
                )


                camera_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )


            cap.release()

    else:

        camera_placeholder.info(
            "Enable **Start Camera** from the sidebar."
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
            use_container_width=True,
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

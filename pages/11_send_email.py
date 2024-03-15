# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
import streamlit as st
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_email_test(to_emails = 'testEmail@gprof.com',
               subject='Test Email: Please respond',
               message = '<strong>This should work</strong> even without Bolding'):
  message = Mail(
      from_email='counsel@gprof.com',
      to_emails=to_emails,
      subject=subject,
      html_content=message)
  try:
      sg = SendGridAPIClient(st.secrets['SENDGRID_API_KEY'])
      response = sg.send(message)
      print(response.status_code)
      print(response.body)
      print(response.headers)
      print("Completed successfully")
  except Exception as e:
      print(e.message)
      print("Did not succeed")
if st.button('send email'):
   send_email_test()
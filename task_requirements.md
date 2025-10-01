Build a multi-tenant conversational sales agent that routes queries to the correct org/branch knowledge base, captures leads if purchase interest is detected, manages bookings with Google Calendar, and sends confirmation emails.

1. Multi-Tenant RAG (Knowledge Base Access)
   - Connect to Pinecone (or another vector DB).
   - Ensure correct knowledge base is selected using orgId + branchId.
   - Example: user from Org 1, Branch A should only query that org/branch’s KB.
2. Conversational Lead Capture
   - Before answering from KB, check if user shows purchase interest.
   - If yes, act like a sales rep and capture:
     a) Product(s) they are interested in
     b) Why they are interested
     c) Their budget/price expectations
     d) Their details(name, email, phone) in very conversational way
   - Save lead information into a MongoDB database.
3. Appointment Management
   - Allow booking, rescheduling, and cancelling appointments.
   - Sync all bookings with Google Calendar (per org/branch calendar).
   - Send confirmation email after booking.
4. Architecture & Orchestration
   - Use LangGraph (or similar) to structure conversation flow:
     a) Intent detection (purchase, info, booking, etc.)
     b) Lead capture flow
     c) Retrieval-Augmented Generation from Pinecone
     d) Booking workflow with Google Calendar + email integration

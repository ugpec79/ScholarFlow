import csv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from papers.views import get_recommendations_data  # Assuming this is the function to fetch recommendations
from .models import GuestUser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class GuestRegisterView(APIView):
    def post(self, request):
        user_agent = request.META.get('HTTP_USER_AGENT', 'unknown')

        # Check if a GuestUser with this user_agent already exists
        guest = GuestUser.objects.filter(user_agent=user_agent).first()

        if guest:
            # Return existing guest UUID
            return Response({'uuid': str(guest.uuid)}, status=status.HTTP_200_OK)
        else:
            # Create new guest
            guest = GuestUser.objects.create(user_agent=user_agent)
            return Response({'uuid': str(guest.uuid)}, status=status.HTTP_201_CREATED)





def log_user_action(user_id, paper_id, action, comment=None, file_path="user_actions.csv"):
    assert action in ['LIKE', 'DISLIKE', 'CLICKED', 'COMMENT'], "Invalid action"

    if action == 'COMMENT' and not comment:
        raise ValueError("Comment must be provided for 'COMMENT' action")

    try:
        with open(file_path, 'r', newline='') as f:
            has_header = next(csv.reader(f), None) is not None
    except FileNotFoundError:
        has_header = False

    entry = [user_id, paper_id, action, comment if comment else ""]

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not has_header:
            writer.writerow(['user_id', 'paper_id', 'action', 'comment'])
        writer.writerow(entry)

    print(f"✅ Logged: {user_id} | {paper_id} | {action} | {comment if comment else 'N/A'}")


class LogUserActivityView(APIView):
    def post(self, request):
        try:
            data = request.data

            user_id = data.get('user_id')
            paper_id = data.get('paper_id')
            action = data.get('action')
            comment = data.get('comment', None)

            if not user_id or not paper_id or not action:
                return Response(
                    {"error": "Missing required parameters: user_id, paper_id, and action are required."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Log to CSV
            log_user_action(user_id, paper_id, action, comment)

            # Handle recommendation if clicked
            if action == "CLICKED":
                recommendations_data = get_recommendations_data(paper_id)
                return Response({
                    "message": f"User action '{action}' logged successfully!",
                    "paper_id": paper_id,
                    "paper_details": recommendations_data.get("input_paper", {}),
                    "recommendations": recommendations_data.get("recommendations", [])
                }, status=status.HTTP_200_OK)

            return Response({"message": f"User action '{action}' logged successfully!"}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
